# Data

Both graphs are built from the public PDNS-Net 2-month dataset (QCRI, https://github.com/qcri/PDNS-Net).
Use the environment in `requirements.txt` or `environment.yml`: the build writes CSVs with a fixed line
terminator and a fixed seed, so the outputs are byte-for-byte reproducible. Every run checks the file
hashes pinned in `configs/pdns.yaml` and `configs/bdns.yaml` before use and stops on a mismatch.

## 1. Source

Download PDNS-Net and note the directory that holds `domains.csv`, `ips.csv`, and the
`timestamp_*_edges.csv` files (`<pdns_2m_raw>` below).

## 2. Full graph (`data/DNS_pdns`, 897,635 domains)

    cd MintA
    python tools/build_pdns.py <pdns_2m_raw> data/DNS_pdns --ip-prune 1500

## 3. Subset (`data/DNS_bdns`, 7,495 domains: 4,668 benign, 2,827 malicious)

    cd MintA
    python tools/build_subset.py data/DNS_pdns data/DNS_bdns --benign 4668 --malicious 2827 --seed 20240520

## 4. Check

    cd MintA
    sha256sum data/DNS_pdns/* data/DNS_bdns/*
    python -m minta check-config --config configs/pdns.yaml

The expected SHA-256 values are listed under `data.expected_sha256` in the two configs.
