#!/usr/bin/env nextflow
/*
 * RNA-seq Pipeline — Nextflow DSL2
 * ===================================
 * FASTQ → QC → Trim → Align → Count → MultiQC
 *
 * Usage:
 *   nextflow run rnaseq_pipeline.nf -profile local
 *   nextflow run rnaseq_pipeline.nf -profile slurm
 *   nextflow run rnaseq_pipeline.nf -profile aws
 *
 * Parameters: see nextflow.config or pass with --param value
 */

nextflow.enable.dsl = 2

// ── Parameters ────────────────────────────────────────────────────────────────

params.reads        = "${baseDir}/data/raw_reads/*_{R1,R2}.fastq.gz"
params.genome       = "${baseDir}/references/hg38.fa"
params.gtf          = "${baseDir}/references/gencode.v44.annotation.gtf"
params.star_index   = "${baseDir}/references/hg38_star_index"
params.outdir       = "${baseDir}/results_nextflow"
params.paired       = true
params.strandedness = 2          // featureCounts: 0=unstranded, 1=sense, 2=antisense
params.min_length   = 36
params.min_quality  = 20
params.max_mismatch = 20

// ── Helper: log pipeline info ─────────────────────────────────────────────────

log.info """
    ╔══════════════════════════════════════╗
    ║     RNA-seq Aging Pipeline v1.0      ║
    ╚══════════════════════════════════════╝
    reads        : ${params.reads}
    genome       : ${params.genome}
    gtf          : ${params.gtf}
    star_index   : ${params.star_index}
    outdir       : ${params.outdir}
    paired       : ${params.paired}
    strandedness : ${params.strandedness}
    """.stripIndent()

// ── Input Channel ─────────────────────────────────────────────────────────────

Channel
    .fromFilePairs(params.reads, checkIfExists: true)
    .set { raw_reads_ch }

// ═══════════════════════════════════════════════════════════════════════════════
// PROCESSES
// ═══════════════════════════════════════════════════════════════════════════════

process FASTQC {
    /*
     * Quality control of raw reads
     */
    tag          "${sample_id}"
    label        "low_cpu"
    publishDir   "${params.outdir}/fastqc", mode: "copy"
    conda        "envs/rnaseq.yaml"

    input:
    tuple val(sample_id), path(reads)

    output:
    path("*.html"), emit: html
    path("*.zip"),  emit: zip

    script:
    """
    fastqc --threads ${task.cpus} --outdir . ${reads}
    """
}


process TRIM_GALORE {
    /*
     * Adapter trimming and quality filtering
     */
    tag          "${sample_id}"
    label        "medium_cpu"
    publishDir   "${params.outdir}/trimmed", mode: "copy"
    conda        "envs/rnaseq.yaml"

    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("*{_trimmed,_val_1,_val_2}.fq.gz"), emit: trimmed_reads
    path("*trimming_report.txt"),                     emit: trimming_report

    script:
    def paired = params.paired ? "--paired" : ""
    """
    trim_galore \\
        ${paired} \\
        --quality ${params.min_quality} \\
        --length  ${params.min_length} \\
        --cores   ${task.cpus} \\
        --gzip \\
        ${reads}
    """
}


process STAR_INDEX {
    /*
     * Build STAR genome index (skip if already built)
     */
    label    "high_cpu"
    conda    "envs/rnaseq.yaml"
    storeDir "${params.star_index}"

    input:
    path(genome_fasta)
    path(gtf)

    output:
    path("*")

    when:
    !file(params.star_index).exists()

    script:
    """
    STAR \\
        --runMode genomeGenerate \\
        --runThreadN ${task.cpus} \\
        --genomeDir . \\
        --genomeFastaFiles ${genome_fasta} \\
        --sjdbGTFfile ${gtf} \\
        --genomeSAindexNbases 14
    """
}


process STAR_ALIGN {
    /*
     * Two-pass STAR alignment to reference genome
     */
    tag          "${sample_id}"
    label        "high_cpu"
    publishDir   "${params.outdir}/star/${sample_id}", mode: "copy"
    conda        "envs/rnaseq.yaml"

    input:
    tuple val(sample_id), path(trimmed_reads)
    path(star_index)

    output:
    tuple val(sample_id), path("Aligned.sortedByCoord.out.bam"), emit: bam
    path("Log.final.out"),                                        emit: star_log
    path("SJ.out.tab"),                                           emit: sj

    script:
    def paired_reads = params.paired ?
        "${trimmed_reads[0]} ${trimmed_reads[1]}" :
        "${trimmed_reads[0]}"
    """
    STAR \\
        --runThreadN ${task.cpus} \\
        --genomeDir ${star_index} \\
        --readFilesIn ${paired_reads} \\
        --readFilesCommand zcat \\
        --outFileNamePrefix ./ \\
        --outSAMtype BAM SortedByCoordinate \\
        --outSAMattributes NH HI AS NM MD \\
        --quantMode TranscriptomeSAM \\
        --twopassMode Basic \\
        --outFilterMultimapNmax ${params.max_mismatch} \\
        --alignSJoverhangMin 8 \\
        --alignMatesGapMax 1000000 \\
        --alignIntronMax 1000000
    """
}


process SAMTOOLS_INDEX {
    /*
     * Index BAM files for fast access
     */
    tag   "${sample_id}"
    conda "envs/rnaseq.yaml"

    input:
    tuple val(sample_id), path(bam)

    output:
    tuple val(sample_id), path(bam), path("${bam}.bai")

    script:
    """
    samtools index -@ ${task.cpus} ${bam}
    """
}


process FEATURECOUNTS {
    /*
     * Count reads per gene across all samples
     */
    label        "medium_cpu"
    publishDir   "${params.outdir}/counts", mode: "copy"
    conda        "envs/rnaseq.yaml"

    input:
    path(bams)
    path(bai_files)
    path(gtf)

    output:
    path("counts.txt"),         emit: counts
    path("counts.txt.summary"), emit: summary

    script:
    def paired = params.paired ? "-p --countReadPairs" : ""
    """
    featureCounts \\
        -T ${task.cpus} \\
        -a ${gtf} \\
        -o counts.txt \\
        ${paired} \\
        -s ${params.strandedness} \\
        -t exon \\
        -g gene_id \\
        -M --primary \\
        ${bams.join(' ')}
    """
}


process MULTIQC {
    /*
     * Aggregate all QC reports into a single HTML report
     */
    publishDir "${params.outdir}/multiqc", mode: "copy"
    conda      "envs/rnaseq.yaml"

    input:
    path(qc_files)

    output:
    path("multiqc_report.html")
    path("multiqc_data/")

    script:
    """
    multiqc . --filename multiqc_report.html --force
    """
}


process FORMAT_MATRIX {
    /*
     * Format featureCounts output into a clean CSV count matrix
     */
    publishDir "${params.outdir}/counts", mode: "copy"

    input:
    path(counts)

    output:
    path("count_matrix.csv")

    script:
    """
    python3 - <<'EOF'
import pandas as pd
from pathlib import Path

df = pd.read_csv("${counts}", sep="\\t", comment="#")
count_cols = [c for c in df.columns if "Aligned" in c or ".bam" in c]
sample_names = [Path(c).stem.replace(".Aligned.sortedByCoord.out","") for c in count_cols]

df_out = df[["Geneid"] + count_cols].copy()
df_out.columns = ["gene_id"] + sample_names
df_out.to_csv("count_matrix.csv", index=False)
print(f"Matrix: {len(df_out)} genes x {len(sample_names)} samples")
EOF
    """
}

// ═══════════════════════════════════════════════════════════════════════════════
// WORKFLOW
// ═══════════════════════════════════════════════════════════════════════════════

workflow {

    // ── QC of raw reads
    FASTQC(raw_reads_ch)

    // ── Trimming
    TRIM_GALORE(raw_reads_ch)

    // ── STAR index (build if missing)
    star_index_ch = Channel.fromPath(params.star_index, checkIfExists: false)

    // ── Alignment
    STAR_ALIGN(TRIM_GALORE.out.trimmed_reads, star_index_ch)

    // ── BAM indexing
    SAMTOOLS_INDEX(STAR_ALIGN.out.bam)

    // ── Quantification
    bam_ch  = SAMTOOLS_INDEX.out.map { sid, bam, bai -> bam }.collect()
    bai_ch  = SAMTOOLS_INDEX.out.map { sid, bam, bai -> bai }.collect()

    FEATURECOUNTS(bam_ch, bai_ch, Channel.fromPath(params.gtf))

    // ── Format count matrix
    FORMAT_MATRIX(FEATURECOUNTS.out.counts)

    // ── MultiQC aggregation
    qc_ch = FASTQC.out.zip
        .mix(STAR_ALIGN.out.star_log)
        .mix(FEATURECOUNTS.out.summary)
        .collect()

    MULTIQC(qc_ch)
}

// ── Completion handler ────────────────────────────────────────────────────────

workflow.onComplete {
    log.info(
        workflow.success
        ? "\nPipeline completed successfully!\nResults: ${params.outdir}"
        : "\nPipeline FAILED. Check logs for details."
    )
}
