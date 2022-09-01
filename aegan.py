#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:09:49 2021

@author: mingjing.xu
"""

from torch import nn
import torch
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MLPblock(nn.Module):
    """
    MLPblock class to create MLP block.
    """
    def __init__(self, in_feat, out_feat, leaky_rate=0.2, normalize=True):
        super(MLPblock, self).__init__()
        
        self.linear = nn.Linear(in_feat, out_feat)
        self.bn = nn.BatchNorm1d(out_feat)
        self.leaky_relu = nn.LeakyReLU(leaky_rate, inplace=True)
        
        block = [self.linear]
        if normalize:
            block.append(self.bn)
        block.append(self.leaky_relu)
        
        self.net = nn.Sequential(*block)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
    
class CNN1dblock(nn.Module):
    """
    CNN1dblock class to create CNN1d block.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0,
                 dilation=1, bias=True, 
                 leaky_rate=0.2, normalize=True):
        super(CNN1dblock, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, 
                                dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(leaky_rate, inplace=True)
        
        block = [self.conv1d]
        if normalize:
            block.append(self.bn)
        block.append(self.leaky_relu)
        
        self.net = nn.Sequential(*block)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
    
class CNNTranspose1dblock(nn.Module):
    """
    CNNTranspose1dblock class to generate CNN transpose 1d block.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0,
                 dilation=1, bias=True, 
                 leaky_rate=0, normalize=True):
        super(CNNTranspose1dblock, self).__init__()
        
        self.convtranpose1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=padding, 
                                dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky_relu = nn.LeakyReLU(leaky_rate, inplace=True)
        
        block = [self.convtranpose1d]
        if normalize:
            block.append(self.bn)
        block.append(self.leaky_relu)
        
        self.net = nn.Sequential(*block)
    
    def forward(self, x):
        out = self.net(x)
        return out
    

class Generator(nn.Module):
    """
    Generator class of Generative Adversarial Network.
    """
    def __init__(self, latent_dim=16, block='mlp', nf=50):
        super(Generator, self).__init__()
        
        if block == 'cnn':
            self.model = nn.Sequential(
                CNNTranspose1dblock(in_channels=latent_dim, out_channels=128, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=True),
                CNNTranspose1dblock(in_channels=128, out_channels=64, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=True),
                CNNTranspose1dblock(in_channels=64, out_channels=32, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=True),
                CNNTranspose1dblock(in_channels=32, out_channels=16, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=True),
                nn.ConvTranspose1d(in_channels=16, out_channels=3, kernel_size=4,
                          stride=2, padding=0, bias=False),
                nn.Tanh()
                )
        elif block == 'mlp':
            self.model = nn.Sequential(
                MLPblock(in_feat=latent_dim, out_feat=64, leaky_rate=0, normalize=False),
                MLPblock(in_feat=64, out_feat=128, leaky_rate=0, normalize=False),
                MLPblock(in_feat=128, out_feat=256, leaky_rate=0, normalize=False),
                nn.Linear(in_features=256, out_features=nf),
                nn.Tanh()
                )
        
    def forward(self, z):
        return self.model(z)
    

class Discriminator(nn.Module):
    """
    Discriminator class of Generative Adversarial Network.
    """
    def __init__(self, block='mlp', nf=50):
        super(Discriminator, self).__init__()
        
        if block == 'cnn':
            self.model = nn.Sequential(
                CNN1dblock(in_channels=3, out_channels=16, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0.2, normalize=False),
                CNN1dblock(in_channels=16, out_channels=32, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0.2, normalize=True),
                CNN1dblock(in_channels=32, out_channels=64, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0.2, normalize=True),
                CNN1dblock(in_channels=64, out_channels=128, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0.2, normalize=True),
                nn.Conv1d(in_channels=128, out_channels=1, kernel_size=4,
                          stride=1, padding=0, bias=False),
                nn.Sigmoid()
                )   
        elif block == 'mlp':
            self.model = nn.Sequential(
                MLPblock(in_feat=nf, out_feat=256, leaky_rate=0.2, normalize=False),
                MLPblock(in_feat=256, out_feat=128, leaky_rate=0.2, normalize=False),
                MLPblock(in_feat=128, out_feat=64, leaky_rate=0.2, normalize=False),
                nn.Linear(in_features=64, out_features=1),
                nn.Sigmoid()
                )
        
    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    """
    Encoder class of Auto Encoder.
    """
    def __init__(self, latent_dim=16, block='mlp', nf=50):
        super(Encoder, self).__init__()
        
        if block == 'cnn':
            self.model = nn.Sequential(
                CNN1dblock(in_channels=3, out_channels=16, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=False),
                CNN1dblock(in_channels=16, out_channels=32, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=True),
                CNN1dblock(in_channels=32, out_channels=64, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=True),
                CNN1dblock(in_channels=64, out_channels=128, kernel_size=4,
                          stride=2, padding=0, bias=False,
                          leaky_rate=0, normalize=True),
                nn.Flatten(),
                nn.Linear(in_features=128*4, out_features=latent_dim, bias=False)
                )   
        elif block == 'mlp':
            self.model = nn.Sequential(
                MLPblock(in_feat=nf, out_feat=256, leaky_rate=0.2, normalize=False),
                MLPblock(in_feat=256, out_feat=128, leaky_rate=0.2, normalize=False),
                MLPblock(in_feat=128, out_feat=64, leaky_rate=0.2, normalize=False),
                nn.Linear(in_features=64, out_features=latent_dim)
                )
        
    def forward(self, x):
        return self.model(x)
        
    
def _pairwise_distances_no_broadcast_helper(X, Y):  
    """
    Calculate the pairwise distance between X and Y.

    Parameters
    ----------
    X : numpy.array (n_samples, n_features)
        First input samples.
    Y : numpy.array (n_samples, n_features)
        Second input samples.
    Returns
    -------
    distance : numpy.array (n_samples,)
        pairwise distance array.
    """
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


def _add_adaptive_noise(X, gamma=0.02, delta=0.001): 
    """
    Add adaptive noise :math:`\epsilon` to the input :math:`X`. 
    
    .. math::

        \epsilon = \gamma * \mathrm{std}(X) + \delta
    
    where :math:`\gamma` is a random variable distributed by Gaussian `N(0, gamma)`,
    :math:`\delta` is a random variable distributed by Gaussian `N(0,delta)`.
    
    Parameters
    ----------
    X : numpy.array (n_samples, n_features)
        First input samples.
    gamma : float, optional
        the parameter of multiplication factor.
    delta : float, optional
        the parameter of addition factor, in case that :math:`\mathrm{std}(X)=0`.

    Returns
    -------
    X_noise : numpy.array (n_samples, n_features)

    """
    sigma = gamma * X.std(axis=0) + delta
    n_samples, n_features = X.shape[0], X.shape[1]
    noise = np.random.randn(n_samples, n_features)
    adaptive_noise = X + noise * np.tile(sigma, (n_samples, 1))
    return adaptive_noise


class AEGAN(object):
    """
    Auto Encoder and Generative Adversarial Network (GAN) are combined to do the anomaly detection. 
    GAN is composed by `G()` network and `D()` network, which generates the distribution of real data, where :math:`G(z)` map the latent variable :math:`z` distributed by Gaussian `N(0,1)` into real data space :math:`x \in \mathcal{X}`. Encoder and Generator are combined into Auto-Encoder and define the reconstruction error 
    
    .. math::

                                 ||(x - G(E(x)))||_2
    
    as the anomaly score.

    
    Parameters
    ----------
    nf : int
        the dimension of data `x`. 
    latent_dim : int, optional(default=16)
        latent variable :math:`z` dimension. 
    batch_size : int, optional(default=128)
        dataset batch size, these data batch are used to train GAN and AE. 
    epochs : int, optional(default=100)
        number of epoches to train the network. 
    k_step_D : int, optional(default=5)
        the number of `D()` network updating times for each `G()` network updating. 
    lr : float, optional(default=0.0002)
        the learning rate for the AEGAN training. 
    block : str, optional(default='mlp')
        `E()` , `G()` , `D()` network base module, 'mlp' or 'cnn'.
    weight_decay : float, optional(default=1e-4)
        weight decay parameters for AEGAN training. 
    validation_ratio : float, optional(default=0.2)
        ratio of validation set from the whole training set, designed to obtain the threshold. 
    contamination : float, optional(default=0.01)
        assuming a certain ratio of the samples are abnormal in the dataset, which is used to choose the appropriate threshold.
    gamma : float, optional(default=0.02)
        the parameter of adaptive noise `gamma`, which is the scaling factor of the noise. 
    delta : float, optional(default=0.001)
        the parameter of adaptive noise `delta`, which is the bias factor of the noise.  
    random_state : int, optional(default=0)
        random state.
    device : str, optional(default='cuda')
        the device to run the AEGAN model, 'cuda' or 'cpu'.
    
    Reference
    ---------
    .. [1] ICSRS2019-R0142, Anomaly Detection for Industrial Systems using Generative Adversarial Networks,
         Mingjing XU, Politecnico di Milano, Italy. http://www.icsrs.org/icsrs19.html

    Attribute
    ---------
    threshold_ : float
        the obtained threshold value, the number larger than it will be identified as abnormal.
    decision_scores_ : List[float]
        the decision scores of training set (anomaly score or reconstruction error).
    generator : nn.Module
        `G()` network module
    discriminator : nn.Module
        `D()` network module
    encoder : nn.Module
        `E()` network module
    JSD_div_ : List[float]
        Jensen-Shannon divergence array in GAN training.
    rec_loss_ : List[float]
        reconstruction error during Auto-Encoder network training. 
    
    Note
    ----
    - JSD approximate 0 means GAN has good performance, also, the AEGAN.
    - rec_loss_ is larger means data is difficult to be reconstruct and thus the sample is more likely an abnormal sample.
    - model can automatically normalize the input.
    
    Examples
    --------
    >>> X = np.random.randn(100,5)
    >>> # AEGAN() example 1
    >>> aegan = AEGAN(latent_dim=3, batch_size=32, epochs=20,
    ...           k_step_D=5, lr=0.0002, contamination=0.001,
    ...           gamma=0.05, delta=0.001, block='mlp', nf=X.shape[-1], device='cuda')
    >>> # AEGAN() example 2
    >>> aegan = AEGAN(block='mlp', nf=X.shape[-1], device='cpu')
    >>> aegan.fit(X)
    >>> print(aegan.threshold_) # print the threshold_
    >>> print(aegan.decision_scores_) # print the decision_scores_
    >>> X_test = np.random.randn(20,5)
    >>> print(aegan.decision_function(X_test) > aegan.threshold_) # anomaly detection results in test set.
    
    """
    
    def __init__(self, nf, latent_dim=16, batch_size=128, epochs=100,
                 k_step_D=5, lr=0.0002,
                 block='mlp', weight_decay=1e-4,
                 validation_ratio=0.2, contamination=0.01,
                 gamma=0.02, delta=0.001,
                 random_state=0, device='cuda'
                 ):
        super(AEGAN, self).__init__()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.k_step_D = k_step_D
        self.lr = lr
        self.block = block
        self.nf = nf
        self.weight_decay = weight_decay
        self.validation_ratio = validation_ratio
        self.random_state = random_state
        self.contamination = contamination
        self.gamma = gamma
        self.delta = delta
        self.scaler_ = MinMaxScaler(feature_range=(-1, 1))
        self.isTrained = False
        
        self.generator = Generator(latent_dim=latent_dim, block=block, nf=nf).to(device)
        self.discriminator = Discriminator(block=block, nf=nf).to(device)
        self.encoder = Encoder(latent_dim=latent_dim, block=block, nf=nf).to(device)
        
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=weight_decay)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, weight_decay=weight_decay)
        self.e_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr, weight_decay=weight_decay)

        self.bceloss = nn.BCELoss()
        self.mseloss = nn.MSELoss()
        
        self.JSD_div_ = []
        self.rec_loss_ = []

    def _update_GAN(self, batch):
        
        def reset_grad():
            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()
        
        batch_size = self.batch_size
        k_step_D = self.k_step_D
        latent_size = self.latent_dim
        device = self.device
        
        train_samples = batch[0].to(device)  
        
        if len(train_samples.shape) == 3:
            label_shape = torch.Size([batch_size, 1, 1])
            z_var_shape = torch.Size([batch_size, latent_size, 1])
        elif len(train_samples.shape) == 2:
            label_shape = torch.Size([batch_size, 1])
            z_var_shape = torch.Size([batch_size, latent_size])
        
        real_labels = torch.ones(label_shape).to(device)
        fake_labels = torch.zeros(label_shape).to(device)
        
        z = torch.randn(z_var_shape).to(device)
        for i in range(k_step_D):
            d_loss_real = self.bceloss(self.discriminator(train_samples), real_labels)  
            d_loss_fake = self.bceloss(self.discriminator(self.generator(z)), fake_labels) 
            d_loss = d_loss_real + d_loss_fake  
            
            reset_grad()
            d_loss.backward()
            self.d_optimizer.step()
        
        real_score = self.discriminator(train_samples)
        fake_score = self.discriminator(self.generator(z))
        
        z = torch.randn(z_var_shape).to(device)
    
        g_loss = -1 * self.bceloss(self.discriminator(self.generator(z)), fake_labels) 
    
        reset_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        return {'d_loss': d_loss.item(),
                'g_loss': g_loss.item(),
                'D_x': real_score.mean().item(),
                'D_G_z': fake_score.mean().item(),
                'JS_Div_lowerbound': -1./2. * d_loss.item() + math.log(2)}
    
    def _update_Encoder(self, batch):
        
        device = self.device
        true_data = batch[0].to(device)

        generated_data = self.generator(self.encoder(true_data))
        reconstruct_loss = self.mseloss(generated_data, true_data)
        self.e_optimizer.zero_grad()
        reconstruct_loss.backward()
        self.e_optimizer.step()
        
        return {'reconstruct_loss': reconstruct_loss.item()}

    def _process_decision_scores(self):
        """
        Internal function to calculate key attributes:

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """
        self.threshold_ = np.percentile(self.decision_scores_,
                                     100 * (1 - self.contamination),
                                     interpolation='nearest')
        self.labels_ = (self.decision_scores_ > self.threshold_).astype('int').ravel()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    def fit(self, X, y=None):
        """
        AEGAN training function.

        Parameters
        ----------
        X : numpy.array (n_samples, n_features)
            training data.
        y : [type], optional(default=None)
            no use.

        Returns
        -------
        self

        """
        
        X_noise = _add_adaptive_noise(X, gamma=self.gamma, delta=self.delta)
        
        self.scaler_.fit(X_noise)
        X_norm = self.scaler_.transform(X_noise)
        
        split_ind = int(X_norm.shape[0] * (1 - self.validation_ratio))
        np.random.shuffle(X_norm)
        X_train = X_norm[:split_ind]
        X_valid = X_norm[split_ind:]
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train.astype('float32')))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                  shuffle=True, drop_last=True)
        
        for epoch in range(self.epochs):
            for i,batch in enumerate(trainloader, 0):
                log = self._update_GAN(batch)
                self.JSD_div_.append(log['JS_Div_lowerbound'])
                if i % 50 == 0:
                    print('epoch:[{}/{}]  batch:[{}/{}]  JS_div:{}'\
                          .format(epoch+1,self.epochs,i+1,len(train_dataset)//self.batch_size,log['JS_Div_lowerbound']))
                    
        for epoch in range(self.epochs):
            for i,batch in enumerate(trainloader, 0):
                log = self._update_Encoder(batch)
                self.rec_loss_.append(log['reconstruct_loss'])
                if i % 50 == 0:
                    print('epoch:[{}/{}]  batch:[{}/{}]  rec_loss:{}'\
                          .format(epoch+1,self.epochs,i+1,len(train_dataset)//self.batch_size,log['reconstruct_loss']))
        
        X_valid_tensor = torch.from_numpy(X_valid.astype('float32')).to(self.device)
        generated_X_valid_tensor = self.generator(self.encoder(X_valid_tensor))
        
        self.decision_scores_ = _pairwise_distances_no_broadcast_helper(X_valid_tensor.cpu().detach().numpy(),
                                                     generated_X_valid_tensor.cpu().detach().numpy())
        
        self._process_decision_scores()

        self.isTrained = True
        
        return self

    def decision_function(self, X):
        """
        compute the anomaly score.

        Parameters
        ----------
        X : numpy.array (n_samples, n_features)
            test samples

        Returns
        -------
        numpy.array (n_samples,)
            anomaly score of samples.
        """
        
        X_norm = self.scaler_.transform(X)
        
        X_tensor = torch.from_numpy(X_norm.astype('float32')).to(self.device)
        generated_X_tensor = self.generator(self.encoder(X_tensor))
        
        return _pairwise_distances_no_broadcast_helper(X_tensor.cpu().detach().numpy(),
                                                       generated_X_tensor.cpu().detach().numpy())

    def save_to_file(self, path):
        """
        save AEGAN model parameters into file.

        Parameters
        ----------
        path : str
            the path string need to save the model.

        Examples
        --------
        >>> aegan_model.save_to_file(path='AEGAN_trained_model.pth')
        """
        import warnings  

        if self.isTrained:  
            torch.save({
                'nf': self.nf,
                'latent_dim': self.latent_dim,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'k_step_D': self.k_step_D,
                'lr': self.lr,
                'block': self.block,
                'weight_decay': self.weight_decay,
                'validation_ratio': self.validation_ratio,
                'random_state': self.random_state,
                'contamination': self.contamination,
                'gamma': self.gamma,
                'delta': self.delta,
                'JSD_div_': self.JSD_div_,
                'rec_loss_': self.rec_loss_,
                'scaler_': self.scaler_,
                'decision_scores_': self.decision_scores_,
                'threshold_': self.threshold_,
                'labels_': self.labels_,
                '_mu': self._mu,
                '_sigma': self._sigma,
                'isTrained': self.isTrained,
                'device': self.device,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'encoder_state_dict': self.encoder.state_dict(),
                'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                'g_optimizer_state_dict': self.g_optimizer.state_dict(),
                'e_optimizer_state_dict': self.e_optimizer.state_dict(),
                'bceloss': self.bceloss,
                'mseloss': self.mseloss
            }, path)
        else:
            warnings.warn("Cannot save untrained model!", Warning)


def load(path):
    """
    load AEGAN model parameters from file..

    Parameters
    ----------
    path : str
        the path string need to load the model.

    Returns
    -------
    AEGAN() object : AEGAN object

    Note
    ----
    Suport case:
    - GPU training, GPU loading
    - CPU training, CPU loading
    - GPU training, CPU loading

    Example
    -------
    >>> aegan_model = load(path='AEGAN_trained_model.pth')

    """

    try:
        checkpoint = torch.load(path)
    except:
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        checkpoint['device'] = torch.device('cpu')

    ag_model = AEGAN(
        nf=checkpoint['nf'],
        latent_dim=checkpoint['latent_dim'],
        batch_size=checkpoint['batch_size'],
        epochs=checkpoint['epochs'],
        k_step_D=checkpoint['k_step_D'],
        lr=checkpoint['lr'],
        block=checkpoint['block'],
        weight_decay=checkpoint['weight_decay'],
        validation_ratio=checkpoint['validation_ratio'],
        random_state=checkpoint['random_state'],
        contamination=checkpoint['contamination'],
        gamma=checkpoint['gamma'],
        delta=checkpoint['delta'],
        device=checkpoint['device'])

    ag_model.JSD_div_ = checkpoint['JSD_div_']
    ag_model.rec_loss_ = checkpoint['rec_loss_']
    ag_model.scaler_ = checkpoint['scaler_']
    ag_model.decision_scores_ = checkpoint['decision_scores_']
    ag_model.threshold_ = checkpoint['threshold_']
    ag_model.labels_ = checkpoint['labels_']
    ag_model._mu = checkpoint['_mu']
    ag_model._sigma = checkpoint['_sigma']
    ag_model.isTrained = checkpoint['isTrained']

    ag_model.generator.load_state_dict(checkpoint['generator_state_dict'])
    ag_model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    ag_model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    ag_model.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    ag_model.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    ag_model.e_optimizer.load_state_dict(checkpoint['e_optimizer_state_dict'])
    ag_model.bceloss = checkpoint['bceloss']
    ag_model.mseloss = checkpoint['mseloss']

    return ag_model


