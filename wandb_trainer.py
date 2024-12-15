import torch
import wandb
import time
from contextlib import contextmanager
from shared_imports import *
from environment import *
from loss_functions import *

# Usage:
# 1. replace `Trainer` with `WandbTrainer`
# 2. add `use_wandb=True` parameter to `trainer.train`
# 3. install `wandb` and login with `wandb login`

@contextmanager
def timer(name, log_wandb=True, rank=-1, epoch=None):
    """Context manager to measure execution time"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if log_wandb and wandb.run is not None and rank <= 0:
        wandb.log({f"time_{name}": elapsed}, step=epoch)
    return elapsed

class WandbTrainer:
    def __init__(self, device='cpu'):
        self.device = device
        self.wandb_initialized = False

    def init_wandb(self, setting_name, hyperparams_name, problem_params=None, store_params=None, 
                  warehouse_params=None, echelon_params=None, observation_params=None, 
                  trainer_params=None, optimizer_params=None, nn_params=None):
        """Initialize WandB with config parameters"""
        if not self.wandb_initialized:
            config = {
                "setting_name": setting_name,
                "hyperparams_name": hyperparams_name
            }
            
            if problem_params is not None:
                config["problem_params"] = problem_params
            if store_params is not None:
                config["store_params"] = store_params
            if warehouse_params is not None:
                config["warehouse_params"] = warehouse_params
            if echelon_params is not None:
                config["echelon_params"] = echelon_params
            if observation_params is not None:
                config["observation_params"] = observation_params
            if trainer_params is not None:
                config["trainer_params"] = trainer_params
            if optimizer_params is not None:
                config["optimizer_params"] = optimizer_params
            if nn_params is not None:
                config["nn_params"] = nn_params

            wandb.init(
                project="neural-inventory-control",
                name=f"{setting_name}_{hyperparams_name}",
                config=config
            )
            self.wandb_initialized = True

    def extract_config_names(self, trainer_params):
        """Extract setting and hyperparams names from save folders path"""
        if 'save_model_folders' in trainer_params:
            return trainer_params['save_model_folders'][-1], trainer_params['save_model_folders'][-1]
        return "unknown_setting", "unknown_hyperparams"

    def train(self, epochs, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params, use_wandb=False):
        rank = trainer_params.get('rank', -1)
        if use_wandb and rank <= 0:
            setting_name, hyperparams_name = self.extract_config_names(trainer_params)
            self.init_wandb(
                setting_name=setting_name,
                hyperparams_name=hyperparams_name,
                problem_params=problem_params,
                store_params=params_by_dataset.get('store_params'),
                warehouse_params=params_by_dataset.get('warehouse_params'),
                echelon_params=params_by_dataset.get('echelon_params'),
                observation_params=observation_params,
                trainer_params=trainer_params,
                optimizer_params={'learning_rate': optimizer.param_groups[0]['lr']},
                nn_params=model.state_dict() if hasattr(model, 'state_dict') else None
            )        

        with timer("total_training", use_wandb, rank=rank):
            for epoch in range(epochs):
                if 'sampler' in trainer_params:
                    trainer_params['sampler'].set_epoch(epoch)
                
                start_time = time.time()
                # Training phase
                model.train()
                with timer("epoch", use_wandb, rank=rank, epoch=epoch):
                    average_train_loss, average_train_loss_to_report = self.do_one_epoch(
                        optimizer,
                        data_loaders['train'],
                        loss_function,
                        simulator,
                        model,
                        params_by_dataset['train']['periods'],
                        problem_params,
                        observation_params,
                        train=True,
                        ignore_periods=params_by_dataset['train']['ignore_periods']
                    )
 
                end_time = time.time()

                if rank <= 0:
                    # Validation phase
                    if epoch % trainer_params['do_dev_every_n_epochs'] == 0:
                        with timer("validation", use_wandb, rank=rank, epoch=epoch):
                            average_dev_loss, average_dev_loss_to_report = self.do_one_epoch(
                                optimizer,
                                data_loaders['dev'],
                                loss_function,
                                simulator,
                                model,
                                params_by_dataset['dev']['periods'],
                                problem_params,
                                observation_params,
                                train=False,
                                ignore_periods=params_by_dataset['dev']['ignore_periods']
                            )
                    else:
                        _, average_dev_loss_to_report = 0, 0

                    # Print results and log to wandb
                    if epoch % trainer_params['print_results_every_n_epochs'] == 0:
                        print(f'epoch: {epoch + 1}')
                        print(f'Average per-period train loss: {average_train_loss_to_report}')
                        print(f'Average per-period dev loss: {average_dev_loss_to_report}')

                    print(f'time (s): {end_time - start_time}')

                    if use_wandb:
                        wandb.log({
                            "train_loss": average_train_loss_to_report,
                            "dev_loss": average_dev_loss_to_report,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                        }, step=epoch)

    def do_one_epoch(self, optimizer, data_loader, loss_function, simulator, model, periods, problem_params, observation_params, train=True, ignore_periods=0, discrete_allocation=False):
        """Do one epoch of training or testing"""
        epoch_loss = 0
        epoch_loss_to_report = 0
        total_samples = len(data_loader.dataset)
        periods_tracking_loss = periods - ignore_periods

        for i, data_batch in enumerate(data_loader):
            # with timer(f"batch_{i}", wandb.run is not None, ):
            data_batch = self.move_batch_to_device(data_batch)
            
            if train:
                optimizer.zero_grad()

            total_reward, reward_to_report = self.simulate_batch(
                loss_function, simulator, model, periods, problem_params, 
                data_batch, observation_params, ignore_periods, discrete_allocation
            )
            
            epoch_loss += total_reward.item()
            epoch_loss_to_report += reward_to_report.item()
            
            mean_loss = total_reward/(len(data_batch['demands'])*periods*problem_params['n_stores'])
            
            if train and model.trainable:
                mean_loss.backward()
                optimizer.step()

        return (epoch_loss/(total_samples*periods*problem_params['n_stores']), 
                epoch_loss_to_report/(total_samples*periods_tracking_loss*problem_params['n_stores']))

    def simulate_batch(self, loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods=0, discrete_allocation=False):
        """Simulate for an entire batch of data"""
        batch_reward = 0
        reward_to_report = 0

        observation, _ = simulator.reset(periods, problem_params, data_batch, observation_params)
        for t in range(periods):
            observation_and_internal_data = {k: v for k, v in observation.items()}
            observation_and_internal_data['internal_data'] = simulator._internal_data

            action = model(observation_and_internal_data)
            
            if discrete_allocation:
                action = {key: val.round() for key, val in action.items()}

            observation, reward, terminated, _, _ = simulator.step(action)
            total_reward = loss_function(None, action, reward)

            batch_reward += total_reward
            if t >= ignore_periods:
                reward_to_report += total_reward
            
            if terminated:
                break

        return batch_reward, reward_to_report

    def test(self, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, discrete_allocation=False, trainer_params=None, use_wandb=False):
        model.eval()
        with torch.no_grad():
            with timer("complete_test", use_wandb):
                average_test_loss, average_test_loss_to_report = self.do_one_epoch(
                    optimizer,
                    data_loaders['test'],
                    loss_function,
                    simulator,
                    model,
                    params_by_dataset['test']['periods'],
                    problem_params,
                    observation_params,
                    train=False,
                    ignore_periods=params_by_dataset['test']['ignore_periods'],
                    discrete_allocation=discrete_allocation
                )

                if use_wandb:
                    wandb.log({
                        "test_loss": average_test_loss_to_report
                    })

                return average_test_loss, average_test_loss_to_report

    def move_batch_to_device(self, data_batch):
        """Move a batch of data to the device (CPU or GPU)"""
        return {k: v.to(self.device) for k, v in data_batch.items()}

    def load_model(self, model, optimizer, model_path):
        """Load a saved model"""
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.warehouse_upper_bound = checkpoint['warehouse_upper_bound']
        return model, optimizer

    def get_time_stamp(self):
        return int(datetime.datetime.now().timestamp())
    
    def get_year_month_day(self):
        ct = datetime.datetime.now()
        return f"{ct.year}_{ct.month:02d}_{ct.day:02d}"
