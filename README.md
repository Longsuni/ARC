# ARC
Urban Region Embedding via Adaptive Risk-aware Consensus Learning


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
