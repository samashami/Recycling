from utils import *

train_data = Trash("dataset-resized", train=True)
val_data = Trash("dataset-resized", train=False)
test_data = Trash("DatasetTest", train=False)


model = LeNet5(n_classes=10, input_channels=3)  # use pretrained model
model.train()

train_avalanche = AvalancheDataset(train_data)
train_loader = DataLoader(train_avalanche, batch_size=4)

val_avalanche = AvalancheDataset(val_data)
val_loader = DataLoader(val_avalanche, batch_size=4)

test_avalanche = AvalancheDataset(test_data)
test_loader = DataLoader(test_avalanche, batch_size=4)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# training prepare
avg_loss = meter.AverageValueMeter()
cm = meter.ConfusionMeter(6)
previous_loss = 1e100

strategy = EWC(
    model,
    optimizer,
    criterion,
    memory_strength,
    train_mb_size=32,
    train_epochs=80,
    eval_mb_size=32,
    device=device,
    plugins=[eval_plugin],
)


for epoch in range(80):
    strategy.train(train_avalanche)
    strategy.eval(val_avalanche)

torch.save(model.state_dict(), "my_model.pth")


test_losses = []
test_accs = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        data, label = V(images), V(labels)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, preds = outputs.max(1)
        num_correct += (preds == labels).sum()
        num_samples += preds.size(0)
        accuracy = num_correct.item() / num_samples

        test_losses.append(loss.item())
        test_accs.append(accuracy)

print(
    "Test Loss: {:.4f}, Test Acc: {:.4f}".format(
        sum(test_losses) / len(test_losses), sum(test_accs) / len(test_accs)
    )
)
