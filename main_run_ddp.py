import yaml
from wandb_trainer import *
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os


CONFIG_SETTING_FILE = "config_files/settings/one_store_lost.yml"
CONFIG_HYPERPARAMS_FILE = "config_files/policies_and_hyperparams/vanilla_one_store.yml"

SETTING_KEYS = (
    "seeds",
    "test_seeds",
    "problem_params",
    "params_by_dataset",
    "observation_params",
    "store_params",
    "warehouse_params",
    "echelon_params",
    "sample_data_params",
)
HYPERPARAMS_KEYS = ("trainer_params", "optimizer_params", "nn_params")


def train(rank: int):
    with open(CONFIG_SETTING_FILE) as file:
        settings = yaml.safe_load(file)

    with open(CONFIG_HYPERPARAMS_FILE) as file:
        hyperparams = yaml.safe_load(file)

    (
        seeds,
        test_seeds,
        problem_params,
        params_by_dataset,
        observation_params,
        store_params,
        warehouse_params,
        echelon_params,
        sample_data_params,
    ) = [settings[key] for key in SETTING_KEYS]
    trainer_params, optimizer_params, nn_params = [
        hyperparams[key] for key in HYPERPARAMS_KEYS
    ]
    observation_params = DefaultDict(lambda: None, observation_params)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    dataset_creator = DatasetCreator()

    if sample_data_params["split_by_period"]:
        scenario = Scenario(
            periods=None,  # period info for each dataset is given in sample_data_params
            problem_params=problem_params,
            store_params=store_params,
            warehouse_params=warehouse_params,
            echelon_params=echelon_params,
            num_samples=params_by_dataset["train"][
                "n_samples"
            ],  # in this case, num_samples=number of products, which has to be the same across all datasets
            observation_params=observation_params,
            seeds=seeds,
        )

        train_dataset, dev_dataset, test_dataset = dataset_creator.create_datasets(
            scenario,
            split=True,
            by_period=True,
            periods_for_split=[
                sample_data_params[k]
                for k in ["train_periods", "dev_periods", "test_periods"]
            ],
        )
    else:
        scenario = Scenario(
            periods=params_by_dataset["train"]["periods"],
            problem_params=problem_params,
            store_params=store_params,
            warehouse_params=warehouse_params,
            echelon_params=echelon_params,
            num_samples=params_by_dataset["train"]["n_samples"]
            + params_by_dataset["dev"]["n_samples"],
            observation_params=observation_params,
            seeds=seeds,
        )

        train_dataset, dev_dataset = dataset_creator.create_datasets(
            scenario,
            split=True,
            by_sample_indexes=True,
            sample_index_for_split=params_by_dataset["dev"]["n_samples"],
        )

        scenario = Scenario(
            params_by_dataset["test"]["periods"],
            problem_params,
            store_params,
            warehouse_params,
            echelon_params,
            params_by_dataset["test"]["n_samples"],
            observation_params,
            test_seeds,
        )

        test_dataset = dataset_creator.create_datasets(scenario, split=False)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params_by_dataset["train"]["batch_size"],
        shuffle=False,
        sampler=train_sampler,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=params_by_dataset["dev"]["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=params_by_dataset["test"]["batch_size"], shuffle=False
    )
    data_loaders = {"train": train_loader, "dev": dev_loader, "test": test_loader}

    init_tensor = torch.rand(next(iter(train_loader))["initial_inventories"].shape)

    neural_net_creator = NeuralNetworkCreator
    model = neural_net_creator().create_neural_network(
        scenario, nn_params, device=device
    )

    for name, m in model.net.items():
        m(init_tensor)

    model = DDP(model)
    model.trainable = True

    loss_function = PolicyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=optimizer_params["learning_rate"]
    )

    simulator = Simulator(device=device)
    trainer = WandbTrainer(device=device)

    trainer_params["base_dir"] = "saved_models"
    trainer_params["save_model_folders"] = [
        trainer.get_year_month_day(),
        nn_params["name"],
    ]
    trainer_params["sampler"] = train_sampler
    trainer_params["rank"] = rank

    # We will simply name the model with the current time stamp
    trainer_params["save_model_filename"] = trainer.get_time_stamp()

    trainer.train(
        trainer_params["epochs"],
        loss_function,
        simulator,
        model,
        data_loaders,
        optimizer,
        problem_params,
        observation_params,
        params_by_dataset,
        trainer_params,
        use_wandb=True,
    )


def main():
    dist.init_process_group("gloo")
    rank = int(os.environ["LOCAL_RANK"])
    train(rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
