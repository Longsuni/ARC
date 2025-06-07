# ARC
This repo contains the source code and the datasets for the ARC (Adaptive Risk-aware Consensus learning) targeting the problem of urban region embedding. The model utilizes multiple urban data as learning views in order to capture the complex characteristics of urban areas with different semantic characteristics.

We introduce two masked strategies for view reconstruction and design both local and regional levels of masking to better cope with dynamic risk changes, and also incorporate a self-weighted contrastive mechanism in consensus learning. These enable ARC to adapt to quality differences among multiple views and effectively mitigate the issue of representation degradation.

## Quick Start


To train and test the ARC model across different cities, use the following parameters:

- **CITY_NAME**: `NewYork(NY)`, `Chicago(Chi)`, or `SanFrancisco(SF)`
- **TASK_NAME**: `checkIn`, `crime`, or `serviceCall`

Run the training script as follows:

```bash
python train.py --city CITY_NAME --task TASK_NAME
```

For example, to train the model on the Chicago dataset for checkIn prediction:

```bash
python train.py --city Chi --task checkIn
```

### Testing with Pre-trained Embeddings

We provide pre-trained embeddings that can be directly used for testing.

- Use the following script to test the model with pre-trained embeddings.

  ```bash
  python test_model.py --city CITY_NAME --task TASK_NAME
  ```

  For example, to test the model on the Chicago dataset for the `checkIn` task:

  ```bash
  python test_model.py --city Chi --task checkIn
  ```


## Dataset Structure

```
data/
├── data_Chicago/
├── data_NewYork/
├── data_SanFrancisco/
├── tasks_Chicago/
├── tasks_NewYork/
└── tasks_SanFrancisco/
```

## Requirements

- **Python:** 3.11.10
- **Dependencies:** 
  - `torch==2.4.1+cu124`
  - `numpy==1.26.4`
