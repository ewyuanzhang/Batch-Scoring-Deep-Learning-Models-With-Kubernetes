
# Batch Scoring Deep Learning Models With Kubernetes
Based on [Batch Scoring Deep Learning Models With AKS](https://github.com/Azure/Batch-Scoring-Deep-Learning-Models-With-AKS)

## Overview
In this repository, we use the scenario of applying vehicle detection onto a video (collection of images). This architecture can be generalized for any batch scoring with deep learning scenario.

## Design
![Reference Architecture Diagram](https://happypathspublic.blob.core.windows.net/assets/batch_scoring_for_dl/batchscoringdl-aks-architecture-diagram.PNG)

The above architecture works as follows:
1. Upload a video file to storage.
2. The video file will trigger Logic App to send a request to the flask endpoint hosted on one of the nodes of the AKS cluster.
3. That node will first preprocess the video file by splitting the video into individual images and extracting the audio file.
4. That node will then add all images to the Service Bus queue.
5. The other nodes in the AKS cluster are continuously polling the Service Bus queue - as soon as any images are in the queue, it will pull it off the queue and apply style transfer to the image.
6. When all frames have been processed, the images will be stitched back together into a video with the audio file.

## Prerequsites

Local/Working Machine:
- Ubuntu >=16.04LTS (not tested on Mac or Windows)
- (Optional) [NVIDIA Drivers on GPU enabled machine](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux) [Additional ref: [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)]
- [Conda >=4.5.4](https://conda.io/docs/)
- [Docker >=1.0](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce-1) 
- [AzCopy >=7.0.0](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-linux?toc=%2fazure%2fstorage%2ffiles%2ftoc.json)
- [Azure CLI >=2.0](https://docs.microsoft.com/en-us/cli/azure/?view=azure-cli-latest)

Accounts:
- [Azure Subscription](https://azure.microsoft.com/en-us/free/) 
- (Optional) A [quota](https://docs.microsoft.com/en-us/azure/azure-supportability/resource-manager-core-quotas-request) for GPU-enabled VMs

While it is not required, it is also useful to use the [Azure Storage Explorer](https://azure.microsoft.com/en-us/features/storage-explorer/) to inspect your storage account.e `az cli` installed and logged into

## Setup

1. Clone the repo `git clone https://github.com/Azure/Batch-Scoring-Deep-Learning-Models-With-AKS`
2. `cd` into the repo
3. Setup your conda env using the _environment.yml_ file `conda env create -f environment.yml` - this will create a conda environment called __batchscoringdl__
4. Activate your environment `source activate batchscoringdl`
5. Log in to Azure using the __az cli__ `az login`

## Steps
Run throught the following notebooks:
1. [Test the vechicle detection script](/00_test_neural_vehicle_detection.ipynb)
2. [Setup Azure - Resource group, Storage, Service Bus](/01_setup_azure.ipynb).
3. [Test the model locally](./02_local_testing.ipynb)
4. [Create the AKS cluster](./03_create_aks_cluster.ipynb)
5. [Run vehicle detection on the cluster](./04_vehicle_detection_on_aks.ipynb)
6. [Deploy Logic Apps](./05_deploy_logic_app.ipynb)
7. [Clean up](./06_clean_up.ipynb)

## Clean up
To clean up your working directory, you can run the `clean_up.sh` script that comes with this repo. This will remove all temporary directories that were generated as well as any configuration (such as Dockerfiles) that were created during the tutorials. This script will _not_ remove the `.env` file. 

To clean up your Azure resources, you can simply delete the resource group that all your resources were deployed into. This can be done in the `az cli` using the command `az group delete --name <name-of-your-resource-group>`, or in the portal. If you want to keep certain resources, you can also use the `az cli` or the Azure portal to cherry pick the ones you want to deprovision. Finally, you should also delete the service principle using the `az ad sp delete` command. 

All the step above are covered in the final [notebook](./06_clean_up.ipynb).
