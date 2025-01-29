from model.classical_methods.base import classical_methods
from copy import deepcopy
import os.path as ops
import pickle
from model.lib.data import (
    Dataset,
)
from model.utils import (
    get_device
)
import numpy as np
import time
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

class CatBoostMethod(classical_methods):
    def __init__(self, args, is_regression):
        self.args = args
        print(args.config)
        self.is_regression = is_regression
        self.D = None
        self.args.device = get_device()
        self.trlog = {}
        assert(args.cat_policy == 'indices')

    def fit(self, N, C, y, info, train=True, config=None):
        if self.D is None:
            self.D = Dataset(N, C, y, info)
            self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
            self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
            self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
            self.data_format(is_train = True)
        model_config = None
        if config is not None:
            self.reset_stats_withconfig(config)
            model_config = config['model']
        
        if model_config is None:
            model_config = self.args.config['model']

        from catboost import CatBoostClassifier, CatBoostRegressor
        
        cat_features = list(range(self.n_num_features, self.n_num_features + self.n_cat_features))
        if self.is_regression:
            self.model=CatBoostRegressor(**model_config, random_state=self.args.seed, cat_features=cat_features, allow_writing_files=False)
        else:
            self.model=CatBoostClassifier(**model_config, random_state=self.args.seed, cat_features=cat_features, allow_writing_files=False,class_weights={0: 1, 1:4},task_type="GPU",
                           devices=str(self.args.gpu),eval_metric='F1')
            # self.model=CatBoostClassifier(**model_config, random_state=self.args.seed, cat_features=cat_features, allow_writing_files=False,eval_metric= "F1",class_weights={0: 1, 1:4})
        if not train:
            return
        results=[]
        kfold=StratifiedKFold(n_splits=self.args.nfold, random_state=self.args.seed, shuffle=True)
        self.model_list=[]
        tic = time.time()
        for i,(train_index, val_index) in enumerate(kfold.split(self.N['train'], self.y['train'])):
            model= deepcopy(self.model)
            N,C,y={},{},{}
            N['train'], N['val'] = deepcopy(self.N['train'][train_index]), deepcopy(self.N['train'][val_index])
            y['train'], y['val'] = deepcopy(self.y['train'][train_index]), deepcopy(self.y['train'][val_index])        
            if self.C is not None:
                C['train'], C['val'] = deepcopy(self.C['train'][train_index]), deepcopy(self.C['train'][val_index])
                N['train'],N['val'] = np.concatenate([N['train'], C['train'].astype(str)], axis=1), np.concatenate([N['val'], C['val'].astype(str)], axis=1)
            fit_config = deepcopy(self.args.config['fit'])
            fit_config['eval_set'] = (N['val'], y['val'])
            model.fit(N['train'], y['train'],**fit_config)
            if not self.is_regression:
                y_val_pred = model.predict(N['val'])
                result = f1_score(y['val'], y_val_pred)
            else:
                y_val_pred = model.predict(N['val'])
                result = mean_squared_error(y['val'], y_val_pred, squared=False)*self.y_info['std']
            results.append(result)
            self.model_list.append(model)
        self.trlog["best_res"]=np.mean(results)

        time_cost = time.time() - tic
        for i in range (len(self.model_list)):
            with open(ops.join(self.args.save_path , 'best-val-{}-fold-{}.pkl'.format(self.args.seed,i)), 'wb') as f:
                pickle.dump(self.model_list[i], f)
        return time_cost

    def predict(self, N, C, y, info, model_name):
        self.model_list=[]
        for i in range(self.args.nfold):
            with open(ops.join(self.args.save_path , 'best-val-{}-fold-{}.pkl'.format(self.args.seed,i)), 'rb') as f:
                model= pickle.load(f)
            self.model_list.append(model)
        self.data_format(False, N, C, y)
        if self.C_test is None:
            test_data = self.N_test
        elif self.N_test is None:
            test_data = self.C_test.astype(str)
        else:
            test_data = np.concatenate([self.N_test, self.C_test.astype(str)], axis=1)
        test_logits=[]
        for model in self.model_list:
            if self.is_regression:
                test_logit = model.predict(test_data)
            else:
                test_logit = model.predict_proba(test_data)
            test_logits.append(test_logit)
        test_logit = np.mean(test_logits, axis=0)
        # vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        return test_logit