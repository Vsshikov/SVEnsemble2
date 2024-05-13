args <- commandArgs(trailingOnly=TRUE)
locallib <- args[1]

r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)

packages.req <- c(
    "argparser",
    "tidyverse",
    "stringdist",
    "testthat",
    "stringr"
    # "survival"
    )

for (package in packages.req){
    install.packages()
}


if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

packages.bioc.req <- c(
    "StructuralVariantAnnotation", 
    "rtracklayer", 
    "BSgenome.Hsapiens.UCSC.hg38"
)

BiocManager::install(version = "3.19")
for (package in packages.bioc.req){
    BiocManager::install(package)
}

