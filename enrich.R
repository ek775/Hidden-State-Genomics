#!/usr/bin/R
# Install required libraries if not already installed
required_packages <- c("optparse", "clusterProfiler", "enrichplot", "enrichR", "org.Hs.eg.db", "dplyr", "ggplot2")
new_packages <- required_packages[!(required_packages %in% installed.packages(quietly = TRUE)[, "Package"])]
if (length(new_packages)) {
    install.packages(new_packages, quietly = TRUE)
}
if (!("org.Hs.eg.db" %in% installed.packages(quietly = TRUE)[, "Package"])) {
    BiocManager::install("org.Hs.eg.db")
}
library(optparse, quietly = TRUE)
library(clusterProfiler, quietly = TRUE)
library(enrichplot, quietly = TRUE)
library(enrichR, quietly = TRUE)
library(org.Hs.eg.db, quietly = TRUE)
library(dplyr, quietly = TRUE)
library(ggplot2, quietly = TRUE)

# Function to perform enrichment analysis
go_enrich <- function(gene_list, outdir, keytype = "SYMBOL", ont = "BP", pval_cutoff = 0.01) {
    # read genes from text file
    print(paste("Reading gene list from:", gene_list))
    genes <- readLines(gene_list, encoding="UTF-8", warn = FALSE, ok = TRUE)

    # use clusterProfiler to perform GO enrichment analysis
    print(paste("Performing GO enrichment analysis for ontology:", ont))
    ego <- enrichGO(gene = genes,
                    OrgDb = org.Hs.eg.db,
                    keyType = keytype,
                    ont = ont,
                    pvalueCutoff = pval_cutoff)

    # convert results to a data frame and filter for adjusted p-value
    ego_df <- as.data.frame(ego)

    # Check if there are any results before generating plots
    if (nrow(ego_df) > 0) {
        # use enrichplot to visualize the enrichment categories
        print("Generating plots...")
        network = cnetplot(ego, node_label = "category")
        barplot = barplot(ego, showCategory = 10)

        # save the results and plots
        print(paste("Saving results to directory:", outdir))
        if (!dir.exists(outdir)) {
            dir.create(outdir, recursive = TRUE)
        }
        ggsave(file.path(outdir, paste0("go_enrichment_", ont, "_network.png")), plot = network, width = 10, height = 8)
        ggsave(file.path(outdir, paste0("go_enrichment_", ont, "_barplot.png")), plot = barplot, width = 10, height = 8)
    } else {
        print("No significant enrichment results found.")
    }
    write.csv(ego_df, file = file.path(outdir, paste0("go_enrichment_", ont, "_results.csv")), row.names = FALSE)
    return(ego_df)
}




# Define CLI options
option_list <- list(
    make_option(c("-g", "--gene_list"), type = "character", help = "Path to the gene list file", metavar = "character"),
    make_option(c("-o", "--outdir"), type = "character", help = "Output directory", metavar = "character"),
    make_option(c("-k", "--keytype"), type = "character", default = "SYMBOL", help = "Key type (default: SYMBOL)", metavar = "character"),
    make_option(c("-t", "--ont"), type = "character", default = "BP", help = "Ontology (BP, MF, CC; default: BP)", metavar = "character"),
    make_option(c("-p", "--pval_cutoff"), type = "numeric", default = 0.01, help = "P-value cutoff (default: 0.01)", metavar = "numeric")
)

# Parse CLI arguments
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Run the enrichment analysis
if (!is.null(opt$gene_list) && !is.null(opt$outdir)) {
    result <- go_enrich(gene_list = opt$gene_list, outdir = opt$outdir, keytype = opt$keytype, ont = opt$ont, pval_cutoff = opt$pval_cutoff)
} else {
    print_help(opt_parser)
    stop("Both --gene_list and --outdir must be provided.", call. = FALSE)
}