from model.methods.base import Method
import time
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np
from model.utils import (
    Averager
)
from typing import Optional
from torch import nn
from model.lib.data import (
    Dataset,
    data_nan_process,
    data_enc_process,
    data_norm_process,
    data_label_process,
    data_loader_process
)
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=1,is_reg=False,n_bins=20,quantiles=None):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.is_reg=is_reg
        self.n_bins=n_bins
        self.quantiles=quantiles

    def forward(self, projections, targets):
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        if self.is_reg:
            targets=self.discretize(targets)
        projections= F.normalize(projections, p=2, dim=1)
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        # dot_tempered= -torch.cdist(projections,projections, p=2)
        # exp_dot_tempered = torch.exp(dot_tempered / self.temperature)+1e-7
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        cardinality_per_samples=cardinality_per_samples[cardinality_per_samples>0]
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        loss_all=torch.sum(log_prob * mask_combined, dim=1)
        loss_all=loss_all[loss_all>0]
        supervised_contrastive_loss_per_sample = loss_all / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss
    # def discretize(self,value,n_bins=20):
    #     quantiles = (torch.arange(n_bins)/n_bins)[1:].double().to(value.device)
    #     quanti_value=torch.quantile(value,quantiles)
    #     label=torch.bucketize(value,quanti_value)
    #     return label
    def discretize(self,value):
        if self.quantiles is not None:
            label=torch.bucketize(value,self.quantiles)
        else:
            quantiles = (torch.arange(self.n_bins)/self.n_bins)[1:].double().to(value.device)
            quanti_value=torch.quantile(value,quantiles)
            label=torch.bucketize(value,quanti_value)
        return label

class ContraKNNMethod(Method):
    def __init__(self, args, is_regression):
        self.features=None
        super().__init__(args, is_regression)
        assert(args.cat_policy == 'tabr_ohe')

    def construct_model(self, model_config = None):
        from model.models.contra_knn import contra_knn
        if model_config is None:
            model_config = self.args.config['model']
        self.model = contra_knn(
            d_in = self.n_num_features + self.C_features,
            d_num = self.n_num_features,
            d_out = self.d_out,
            **model_config
        ).to(self.args.device)
        self.model.double()
    
    
    def data_format(self, is_train = True, N = None, C = None, y = None):
        if is_train:
            self.N, self.C, self.num_new_value, self.imputer, self.cat_new_value = data_nan_process(self.N, self.C, self.args.num_nan_policy, self.args.cat_nan_policy)
            self.y, self.y_info, self.label_encoder = data_label_process(self.y, self.is_regression)
            self.N, self.C, self.ord_encoder, self.mode_values, self.cat_encoder = data_enc_process(self.N, self.C, self.args.cat_policy)
            self.n_num_features = self.N['train'].shape[1] if self.N is not None else 0
            self.n_cat_features = self.C['train'].shape[1] if self.C is not None else 0      
            self.N, self.normalizer = data_norm_process(self.N, self.args.normalization, self.args.seed)
    
            if self.is_regression:
                self.d_out = 1
            else:
                self.d_out = len(np.unique(self.y['train']))
            self.C_features = self.C['train'].shape[1] if self.C is not None else 0
            self.N, self.C, self.y, self.train_loader, self.val_loader, _ = data_loader_process(self.is_regression, (self.N, self.C), self.y, self.y_info, self.args.device, self.args.batch_size, is_train = True)
            if self.is_regression:
                self.n_bins=self.args.config["general"]["n_bins"]
                quantiles=(torch.arange(self.n_bins)/self.n_bins)[1:].double().to(self.y["train"].device)
                self.quantiles=torch.quantile(self.y["train"],quantiles)
        else:
            N_test, C_test, _, _, _ = data_nan_process(N, C, self.args.num_nan_policy, self.args.cat_nan_policy, self.num_new_value, self.imputer, self.cat_new_value)
            y_test, _, _ = data_label_process(y, self.is_regression, self.y_info, self.label_encoder)
            N_test, C_test, _, _, _ = data_enc_process(N_test, C_test, self.args.cat_policy, None, self.ord_encoder, self.mode_values, self.cat_encoder)
            N_test, _ = data_norm_process(N_test, self.args.normalization, self.args.seed, self.normalizer)
            _, _, _, self.test_loader, _ =  data_loader_process(self.is_regression, (N_test, C_test), y_test, self.y_info, self.args.device, self.args.batch_size, is_train = False)
            if N_test is not None and C_test is not None:
                self.N_test,self.C_test = N_test['test'],C_test['test']
            elif N_test is None and C_test is not None:
                self.N_test,self.C_test = None,C_test['test']
            else:
                self.N_test,self.C_test = N_test['test'],None
            self.y_test = y_test['test']

    def fit(self, N, C, y, info, train = True, config = None):
        # if the method already fit the dataset, skip these steps (such as the hyper-tune process)
        if self.D is None:
            self.D = Dataset(N, C, y, info)
            self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
            self.is_binclass, self.is_multiclass, self.is_regression = self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
            self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features
            
            self.data_format(is_train = True)
        if config is not None:
            self.reset_stats_withconfig(config)
        self.construct_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.config['training']['lr'], 
            weight_decay=self.args.config['training']['weight_decay']
        )
        print(self.args.config)
        if self.is_regression:
            self.criterion=SupervisedContrastiveLoss(temperature=self.args.config["model"]["temperature"],is_reg=self.is_regression,n_bins=self.n_bins,quantiles=self.quantiles)
        else:
            self.criterion=SupervisedContrastiveLoss(temperature=self.args.config["model"]["temperature"],is_reg=self.is_regression)
        # if not train, skip the training process. such as load the checkpoint and directly predict the results
        if not train:
            return
        time_cost = 0
        for epoch in range(self.args.max_epoch):
            tic = time.time()
            self.train_epoch(epoch)
            self.validate(epoch)
            elapsed = time.time() - tic
            time_cost += elapsed
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
            if not self.continue_training:
                break
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )
        return time_cost


    def predict(self, N, C, y, info, model_name):
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, model_name + '-{}.pth'.format(str(self.args.seed))))['params'])
        print('best epoch {}, best val res={:.4f}'.format(self.trlog['best_epoch'], self.trlog['best_res']))
        ## Evaluation Stage
        self.model.eval()
        
        self.data_format(False, N, C, y)
        
        test_logit, test_label,test_features = [], [],[]
        features,labels=[],[]
        with torch.no_grad():
            for i, (X, y) in enumerate(self.train_loader, 1):
                if self.N is not None and self.C is not None:
                    X = torch.cat([X[0], X[1]],dim=1)
                feature =self.model(X)
                features.append(feature)
                labels.append(y)
            self.features=torch.cat(features,0)
            self.labels=torch.cat(labels,0)            
            for i, (X, y) in tqdm(enumerate(self.test_loader)):
                if self.N is not None and self.C is not None:
                    X = torch.cat([X[0], X[1]],dim=1)                           
                pred,feature=self.model.predict(X,self.features,self.labels)
                test_logit.append(pred)
                test_label.append(y)
                
                test_features.append(feature)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        test_features=torch.cat(test_features,0)
        vl = self.criterion(test_features, test_label).item()     

        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        # print('Test: loss={:.4f}'.format(vl))
        for name, res in zip(metric_name, vres):
            print('[{}]={:.4f}'.format(name, res))
        return vl, vres, metric_name, test_logit

    def train_epoch(self, epoch):
        self.model.train()
        tl = Averager()
        features,labels=[],[]
        for i, (X, y) in enumerate(self.train_loader, 1):
            self.train_step = self.train_step + 1
            if self.N is not None and self.C is not None:
                X = torch.cat([X[0], X[1]],dim=1)
            feature =self.model(X)
            loss = self.criterion(feature,y)
            features.append(feature.detach())
            labels.append(y)
            tl.add(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i-1) % 50 == 0 or i == len(self.train_loader):
                print('epoch {}, train {}/{}, loss={:.4f} lr={:.4g}'.format(
                    epoch, i, len(self.train_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
            del loss
        self.features=torch.cat(features,0)
        self.labels=torch.cat(labels,0)
        tl = tl.item()
        self.trlog['train_loss'].append(tl)   


    def validate(self, epoch):
        print('best epoch {}, best val res={:.4f}'.format(
            self.trlog['best_epoch'], 
            self.trlog['best_res']))
        
        ## Evaluation Stage
        self.model.eval()
        test_logit, test_label,test_features= [], [],[]
        with torch.no_grad():
            for i, (X, y) in tqdm(enumerate(self.val_loader)):
                if self.N is not None and self.C is not None:
                    X = torch.cat([X[0], X[1]],dim=1)                           
                pred,feature =self.model.predict(X,self.features,self.labels)
                test_logit.append(pred)
                test_label.append(y)
                test_features.append(feature)
                
        test_logit = torch.cat(test_logit, 0)
        test_label = torch.cat(test_label, 0)
        test_features=torch.cat(test_features, 0)
        
        vl = self.criterion(test_features, test_label).item()         

        if self.is_regression:
            task_type = 'regression'
            measure = np.less_equal
        else:
            task_type = 'classification'
            measure = np.greater_equal
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)
        print('epoch {}, val, loss={:.4f} {} result={:.4f}'.format(epoch, vl, task_type, vres[0]))
        if measure(vres[0], self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres[0]
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )
            self.val_count = 0
        else:
            self.val_count += 1
            if self.val_count > 20:
                self.continue_training = False
        torch.save(self.trlog, osp.join(self.args.save_path, 'trlog'))