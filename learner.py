from gnnfree.managers.learner import SingleModelLearner

class GraphTextCCALearner(SingleModelLearner):
    def load(self, batch, device):
        batch.to_device(device)
        batch.to_name()
        batch.graph = self.data.g.to(device)
        return batch

    def forward_func(self, batch):
        input1 = [batch.text]
        input2 = [batch.graph, batch.node_ind]
        output = self.model(input1, input2)
        return output

    def loss_fn(self, res, batch):
        return self.loss(res)

class GraphForwardLearner(SingleModelLearner):
    def load(self, batch, device):
        batch.to_device(device)
        batch.to_name()
        batch.graph = self.data.g.to(device)
        return batch

    def forward_func(self, batch):
        input = [batch.graph, batch.node_ind]
        return self.model(input)

    def loss_fn(self, res, batch):
        return self.loss(res.view(-1), batch.labels)

class ForwardLearner(SingleModelLearner):
    def load(self, batch, device):
        batch.to_device(device)
        batch.to_name()
        return batch

    def forward_func(self, batch):
        input = [batch.text]
        return self.model(input)

    def loss_fn(self, res, batch):
        return self.loss(res, batch.labels)

class MLPForwardLearner(ForwardLearner):
    def setup_optimizer(self, optimizer_groups):
        parameters = [p for p in self.model.mlp.parameters()]
        optimizer_groups[0]['params'] = parameters
        self.optimizer = self.optimizer_type(optimizer_groups)

class GraphForwardLearner(SingleModelLearner):
    def load(self, batch, device):
        batch.to_device(device)
        batch.to_name()
        batch.graph = self.data.g.to(device)
        return batch

    def forward_func(self, batch):
        input = [batch.graph, batch.node_ind]
        return self.model(input)

    def loss_fn(self, res, batch):
        return self.loss(res, batch.labels)

class MLPGraphForwardLearner(GraphForwardLearner):
    def setup_optimizer(self, optimizer_groups):
        parameters = [p for p in self.model.mlp.parameters()]
        optimizer_groups[0]['params'] = parameters
        self.optimizer = self.optimizer_type(optimizer_groups)