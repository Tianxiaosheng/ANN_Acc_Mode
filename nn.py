import torch
#import plotly.graph_objects as go
#from plotly import graph_objs as go
import matplotlib.pyplot as plt
import numpy as np

N, D_in, H, D_out = 16, 1, 1024, 1

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out))

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(30000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 0:
        print(t, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

xx=x.flatten().numpy()
yy=y.flatten().numpy()

#fig.add_trace(go.Scatter(x=x.flatten().numpy(), y=y.flatten().numpy(), mode="markers"))
plt.plot(xx, yy, ".");


minx = min(list(x.numpy()))
maxx = max(list(x.numpy()))
c = torch.from_numpy(np.linspace(minx, maxx, num=640)).reshape(-1, 1).float()
d = model(c)
cc = c.flatten().numpy()
dd = d.flatten().detach().numpy()
#fig.add_trace(go.Scatter(x=c.flatten().numpy(), y=d.flatten().detach().numpy(), mode="lines"))
plt.plot(cc, dd);

plt.show()
