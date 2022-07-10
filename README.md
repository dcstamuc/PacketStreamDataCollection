# Packet Stream Data Collection

### Deep Sequence Models for Packet Stream Analysis and Early Decisions, LCN 2022.
*Minji Kim, Dongeun Lee, Kookjin Lee, Doowon Kim, Sangman Lee, Jinoh Kim*
The packet stream analysis is essential for the early identification of attack connections while in progress, enabling timely responses to protect system resources. However, there are several challenges for implementing effective analysis, including out-of-order packet sequences introduced due to network dynam-ics andclass imbalancewith a small fraction of attack connections available to characterize. To overcome these challenges, we present two deep sequence models: (i) a bidirectional recurrent structure designed for resilience to out-of-order packets, and (ii) a pre-training-enabled sequence-to-sequence structure designed for better dealing with unbalanced class distributions using self-supervised learning. We evaluate the presented models using a real network dataset created from month-long real traffic traces collected from backbone links with the associated intrusion log. The experimental results support the feasibility of the presented models with up to 94.8% in F1 score with the first five packets (k=5), outperforming baseline deep learning models.

## Datasets
The data consists of packet size information, packet interarrival time, c2s and taxonomy label.

```
MAWILabSep2020/
   ├── 0901_0930_K_3.csv
   ├── 0901_0930_K_5.csv
   ├── 0901_0930_K_10.csv
   └── 0901_0930_K_20.csv
```
Due to limitation of file size, we are sharining the [Google drive link](https://drive.google.com/drive/folders/1iLoW97uCg3tirV0MdnYxqo9CveHMXmYW?usp=sharing
) for dataset.


## Acknowledgements, Usage & License

This work was supported in part by the Texas A&M University Presidential GAR Initiative program. 

If you find our work useful in your research or if you use parts of this code please consider citing our paper:
```
```

© The code and datasets are made available under the GPLv3 License and is available for non-commercial academic purposes.



