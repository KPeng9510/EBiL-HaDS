# [Neurips2024] Advancing Open-Set Domain Generalization Using Evidential Bi-Level Hardest Domain Scheduler

## Introduction of our work

In Open-Set Domain Generalization (OSDG), the model is exposed to both new variations of data appearance (domains) and open-set conditions, where both known and novel categories are present at test time. The challenges of this task arise from the dual need to generalize across diverse domains and accurately quantify category novelty, which is critical for applications in dynamic environments. Recently, meta-learning techniques have demonstrated superior results in OSDG, effectively orchestrating the meta-train and -test tasks by employing varied random categories and predefined domain partition strategies. These approaches prioritize a well-designed training schedule over traditional methods that focus primarily on data augmentation and the enhancement of discriminative feature learning. The prevailing meta-learning models in OSDG typically utilize a predefined sequential domain scheduler to structure data partitions. However, a crucial aspect that remains inadequately explored is the influence brought by strategies of domain schedulers during training. In this paper, we observe that an adaptive domain scheduler benefits more in OSDG compared with prefixed sequential and random domain schedulers. We propose the Evidential Bi-Level Hardest Domain Scheduler (EBiL-HaDS) to achieve an adaptive domain scheduler. This method strategically sequences domains by assessing their reliabilities in utilizing a follower network, trained with confidence scores learned in an evidential manner, regularized by max rebiasing discrepancy, and optimized in a bi-level manner. We verify our approach on three OSDG benchmarks, i.e., PACS, DigitsDG, and OfficeHome. The results show that our method substantially improves OSDG performance and achieves more discriminative embeddings for both the seen and unseen categories, underscoring the advantage of a judicious domain scheduler for the generalizability to unseen domains and unseen categories.

## Instruction for the code running

please first prepare the environment following environment.yml


The PACS dataset can be downloaded via Kaggle. DigitsDG dataset and the OfficeHome dataset are all publicly available online.


Download links of the datasets will be provided after the review procedure. 


## Sample for the results reproduction

please first set the dataset path to the correct path for PACS dataset.

Then execute python test_domain_scheduler.py

The model weight can be downloaded from the following anonymous link, please use this model weight to replace the file path during test time

https://drive.google.com/file/d/1gUI2V0MIYO2y0V0_Br6KfNecbdlE4zEZ/view?usp=sharing



You will get the following results on PACS when using cartoon as the target domain.

![Alt text](test.png)


Thank you!


