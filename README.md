# Packet Stream Data Collection

### WIDE Network Dataset and Deep Sequence Models for Early Attack Identification.*
*Minji Kim, Dongeun Lee, Kookjin Lee, Doowon Kim, Sangman Lee, [Jinoh Kim](https://jinoh-cs.github.io/)*  

The packet stream analysis is essential for the early identification of attack connections while in progress, enabling timely responses to protect system resources. However, there are several challenges for implementing effective analysis, including out-of-order packet sequences introduced due to network dynam-ics andclass imbalancewith a small fraction of attack connections available to characterize. To overcome these challenges, we present two deep sequence models: (i) a bidirectional recurrent structure designed for resilience to out-of-order packets, and (ii) a pre-training-enabled sequence-to-sequence structure designed for better dealing with unbalanced class distributions using self-supervised learning. We evaluate the presented models using a real network dataset created from month-long real traffic traces collected from backbone links with the associated intrusion log. The experimental results support the feasibility of the presented models with up to 94.8% in F1 score with the first five packets (k=5), outperforming baseline deep learning models.

## Deep Sequence Models

The code includes the following four models with the instruction to execute:
1. MLP-based non-sequence model
2. LSTM-based sequence model
3. Bi-directional LSTM model
4. Sequence-to-Sequence Autoencoder model

## Datasets

In this work, we construct packet stream data to develop the function for the early identification of network attacks by combining public traffic traces with corresponding intrusion detection logs. Specifically, we utilize the real network traces and intrusion logs collected from backbone links in Japan [(MAWILab)](http://www.fukuda-lab.org/mawilab/). The traffic trace contains TCP/IP packet header information in a pcap file, while the associated intrusion log is provided in a comma-separated values (CSV) file with the attack information inferred by multiple detectors. Each pcap file is a recording of 15-minute traffic collected on a specific day. We extract flow information from **the 25-day network traffic collected in September 2020**, except five days due to unavailability (3rd, 14th, 27th, 28th, and 29th).  

The dataset consists of **packet size information, packet interarrival time, c2s and taxonomy label**. K means number of packets. (e.g., 0901_0930_K_3.csv has 3 packets on the same flow is measured)

```
MAWILabSep2020/
   ├── 0901_0930_K_3.csv
   ├── 0901_0930_K_5.csv
   ├── 0901_0930_K_10.csv
   └── 0901_0930_K_20.csv
```
Due to limitation of file size, we are sharining the [Google drive link](https://drive.google.com/drive/folders/1iLoW97uCg3tirV0MdnYxqo9CveHMXmYW?usp=sharing
) for dataset.

## Issues

If you have questions about your rights to use, please contact dcs.tamuc@gmail.com

## Acknowledgements, Usage & License

This work was supported in part by the Texas A&M University Presidential GAR Initiative program.  

If you find our work useful in your research or if you use parts of this datasets or code please consider citing our paper:  
```
```

© The code and datasets are made available under the GPLv3 License and is available for non-commercial academic purposes.  



