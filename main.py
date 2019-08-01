from Trainer import Trainer
from config import get_config
from data import train_val_data

def main():
	config, unparsed = get_config()
	data_loader, val_data_loader = train_val_data(batch_size=config.batch_size, shuffle = config.shuffle, num_workers = config.num_workers, valid_size = config.valid_size )
	trainer = Trainer(config, data_loader)
	trainer.train()

if __name__ == '__main__':
	main()
