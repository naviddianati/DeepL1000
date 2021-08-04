'''
Created on Jun 22, 2021

@author: Navid Dianati
'''
import logging

from MOA_L1000 import config, utils, trainer , dataload

logger = logging.getLogger('main')
logging.basicConfig()
logger.setLevel('INFO')


def run_model(params):

    utils.export_params(params)
    data_loader = eval(params.get('data_loader'))
    SEEDS = params.get("SEEDS")
    
    # Load data and perform transformations
    X, y, w, X_holdout, y_holdout = data_loader()
   
    # Instantiate trainer
    tr = trainer.Trainer(**params)
    tr.set_training_data(X, y, w, X_holdout, y_holdout)
    
    for seed in SEEDS:
        utils.seed_all(seed)
        tr.seed = seed
        # Run cross-validation on this fold
        try:
            tr.run_cv(
                **params
                )
        except KeyboardInterrupt:
            logger.info('Training aborted.')
            break


def main():
    for params in [
#         config.get_params_DNN5(),
#         config.get_params_DNN6(),
#         config.get_params_DNN10(),
        # config.get_params_DNN11(),
        config.get_params_DNN15(),
        ]:
        run_model(params)

    
if __name__ == "__main__":
    main()
    
