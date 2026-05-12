FROM continuumio/miniconda3:23.10.0-1

LABEL maintainer="CenAGE Portfolio"
LABEL description="RNA-seq aging pipeline — fully reproducible container"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make wget curl git \
    libz-dev libbz2-dev liblzma-dev \
    libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment specification
COPY envs/rnaseq.yaml /opt/envs/rnaseq.yaml

# Create conda environment
RUN conda env create -f /opt/envs/rnaseq.yaml \
    && conda clean -afy

# Activate environment by default
SHELL ["conda", "run", "-n", "rnaseq", "/bin/bash", "-c"]
ENV PATH /opt/conda/envs/rnaseq/bin:$PATH
ENV CONDA_DEFAULT_ENV rnaseq

# Install R packages (DESeq2, etc.)
RUN R -e "\
    options(repos = c(CRAN='https://cran.rstudio.com/')); \
    install.packages('BiocManager'); \
    BiocManager::install(c('DESeq2','apeglm','clusterProfiler', \
                            'org.Hs.eg.db','fgsea','msigdbr', \
                            'Seurat','SingleCellExperiment', \
                            'edgeR','limma')); \
    install.packages(c('tidyverse','ggplot2','pheatmap','cowplot')); \
    "

# Copy pipeline files
WORKDIR /pipeline
COPY Snakefile .
COPY config.yaml .
COPY rnaseq_pipeline.nf .
COPY nextflow.config .

# Set default command
CMD ["snakemake", "--cores", "1", "--dryrun"]
