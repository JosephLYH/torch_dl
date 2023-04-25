from torch.utils.tensorboard import SummaryWriter

def write_tb(step, writer: SummaryWriter, model, scaler):
    if model:
        for name, param in model.named_parameters():
            writer.add_histogram('/'.join(name.split('.')[1:]), param, step)
            if hasattr(param, 'grad') and param.grad is not None:
                writer.add_histogram('/'.join(name.split('.')[1:]) + '/grad', param.grad, step)

    if scaler:
        if hasattr(scaler, '_scale') and scaler._scale is not None:
            writer.add_scalar('scale', scaler._scale, step)