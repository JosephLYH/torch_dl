from torch.utils.tensorboard import SummaryWriter

def write_tb(step, writer: SummaryWriter, metrics, model, optimizer, scaler):
    if metrics:
        for name, value in metrics.items():
            writer.add_scalar(name, value, step, new_style=True)

    if model:
        for name, param in model.named_parameters():
            writer.add_histogram('/'.join(name.split('.')), param, step)
            if hasattr(param, 'grad') and param.grad is not None:
                writer.add_histogram('/'.join(name.split('.')) + '/grad', param.grad, step)

    if optimizer:
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)

    if scaler:
        if hasattr(scaler, '_scale') and scaler._scale is not None:
            writer.add_scalar('scale', scaler._scale, step)