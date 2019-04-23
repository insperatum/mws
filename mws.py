from builtins import super

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def nograd(x):
    return x.data if torch.is_tensor(x) else x

class EncoderDecoder(nn.Module):
    def sample(self, i, x):
        """
        i: batch of task indices
        x: batch of instances
        For each instance, samples a (c, x) pair from the posterior predictive
        """
        c, c_score = self.encoder(i, x)
        x, x_score = self.decoder(i, c)
        return c, x

class MWS(EncoderDecoder):
    def __init__(self, encoder, decoder, prior, nTasks, frontierSize=10, nUpdates=1, rObjective="sleep"):
        """
        rObjective: "mem" or "sleep" or "mix"
        """
        super().__init__()
        self.nTasks = nTasks
        self.frontierSize = frontierSize
        self.nUpdates = nUpdates
        self.rObjective = rObjective

        self.mixtureComponents = [[] for _ in range(nTasks)]
        self.mixtureWeights = Parameter(torch.zeros(nTasks, frontierSize)) #Unnormalised log-q
        self.mixtureScores = [[] for _ in range(nTasks)] #most recent log joint
        self.nMixtureComponents = Parameter(torch.zeros(nTasks))
        self.t = Parameter(torch.zeros(1))

        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def sample(self, i, x):
        """
        i: batch of task indices
        x: batch of instances
        For each instance, samples a (c, x) pair from the posterior predictive
        """
        if i is None: return super().sample(i, x)

        ix_update = [(ii,xx) for ii,xx in zip(i,x) if len(self.mixtureComponents[ii]) == 0]
        if len(ix_update)>0: 
            i_update = [ii for ii,xx in ix_update]
            x_update = [xx for ii,xx in ix_update]
            self._updateMemory(i_update, x_update, 1)

        idxs = self._getDist(i).sample()
        idxs = idxs.tolist()
        c = [self.mixtureComponents[ii][idx] for ii,idx in zip(i, idxs)]
        x, _ = self.decoder(i, c)
        return c, x

    def forward(self, i, x, annealing=0):
        """
        i: batch of task indicies
        x: batch of indices
        Returns a variational bound (ELBO) on log p(x)
        The gradients train both the generative model and the recognition model
        """

        ### Wake phase ###
        # Update memory
        self._updateMemory(i, x, self.nUpdates)

        
        ### Sleep phase ###
        # Sleep-r update
        if self.encoder is not None and (self.rObjective == "sleep" or self.rObjective == "mix"):
            _c, _ = self.prior(i)
            _x, _ = self.decoder(i, _c)
            _, r_objective = self.encoder(i, nograd(_x), _c)
        else:
            r_objective = 0


        ### Mem phase ###
        dist = self._getDist(i) 
        j = dist.sample()
        c = [self.mixtureComponents[ii][jj] for ii,jj in zip(i,j.tolist())]
        # Mem-r update
        if self.rObjective == "mem" or self.rObjective=="mix":
            _, r_objective_wake = self.encoder(i, x, c) 
            if self.rObjective == "mem": r_objective = r_objective_wake
            else: r_objective = r_objective*0.5 + r_objective_wake*0.5
        # Mem-p update 
        _, prior = self.prior(i, c)
        _, likelihood = self.decoder(i, c, x)
        mixtureScores = prior + likelihood
        for idx,(ii,jj) in enumerate(zip(i,j)):
            self.mixtureScores[ii][jj] = mixtureScores[idx].item()
        lp = dist.log_prob(j)
        kl = lp - prior
        elbo = likelihood - (1-annealing)*kl
        p_objective = elbo + elbo.data*lp
        


        ### Total ###
        objective = p_objective + r_objective
        if annealing==0: objective = objective - objective.data + elbo.data
        else: objective = objective - objective.data + (likelihood - kl).data

        return objective

    def _getDist(self, i):
        weights = self.mixtureWeights[i]
        return Categorical(F.softmax(weights, dim=1))

    def _updateMemory(self, i, x, nUpdates):
        """
        Use samples from encoder to propose updates to memory
        """
        batch_size = len(x)
        task_update_data = {}

        unfilled_idxs = {idx:self.frontierSize - len(self.mixtureScores[i[idx]])
                            for idx in range(batch_size) if len(self.mixtureScores[i[idx]]) < self.frontierSize}
        if len(unfilled_idxs)>0:
            unfilled_i = [i[idx] for idx,num_repeats in unfilled_idxs.items() for _ in range(num_repeats)]
            unfilled_x = [x[idx] for idx,num_repeats in unfilled_idxs.items() for _ in range(num_repeats)]
            unfilled_c, _ = self.encoder(i, x) if self.encoder is not None else self.prior(i)
            for ii,cc,xx in zip(unfilled_i, unfilled_c, unfilled_x):
                if cc not in self.mixtureComponents[ii]:
                    self.mixtureComponents[ii].append(cc)
                    task_update_data[ii]=xx
            
        for iUpdate in range(nUpdates):
            c, _ = self.encoder(i, x) if self.encoder is not None else self.prior(i)
            _, priorscore = self.prior(i, c)
            _, likelihood = self.decoder(i, c, x)
            score = (priorscore + likelihood).tolist()
            for idx in range(batch_size):
                if (len(self.mixtureScores[i[idx]]) < self.frontierSize or score[idx] > min(self.mixtureScores[i[idx]])) \
                        and c[idx] not in self.mixtureComponents[i[idx]]:
                    self.mixtureComponents[i[idx]].append(c[idx])
                    task_update_data[i[idx]] = x[idx]

        for ii,xx in task_update_data.items():
            _, priorscores = self.prior([ii for _ in self.mixtureComponents[ii]],
                                        self.mixtureComponents[ii])
            _, likelihoods = self.decoder([ii for _ in self.mixtureComponents[ii]],
                                          self.mixtureComponents[ii],
                                          [xx for _ in range(len(self.mixtureComponents[ii]))])
            self.mixtureScores[ii] = (priorscores+likelihoods).tolist()
            while len(self.mixtureComponents[ii]) > self.frontierSize:
                min_idx = np.argmin(self.mixtureScores[ii])
                self.mixtureComponents[ii] = self.mixtureComponents[ii][:min_idx] + self.mixtureComponents[ii][min_idx+1:]
                self.mixtureScores[ii] = self.mixtureScores[ii][:min_idx] + self.mixtureScores[ii][min_idx+1:]
            self.mixtureWeights.data[ii][:len(self.mixtureScores[ii])] = self.t.new(self.mixtureScores[ii])
            self.mixtureWeights.data[ii][len(self.mixtureScores[ii]):] = float("-inf")
            self.nMixtureComponents[ii] = len(self.mixtureComponents[ii])

    def _ensure_nonempty(self, i, x):
        ix_update = [(ii,xx) for ii,xx in zip(i,x) if len(self.mixtureComponents[ii]) == 0]
        if len(ix_update)>0: 
            i_update = [ii for ii,xx in ix_update]
            x_update = [xx for ii,xx in ix_update]
            self._updateMemory(i_update, x_update, 1)
