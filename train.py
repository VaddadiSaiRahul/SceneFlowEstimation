def scheduler(epoch, lr):
    if epoch < 150:
        return lr
    else:
        if epoch % 100 == 0:
            return lr / 2

def train(x_train, y_train, batch_size, model, callbacks):
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=10, verbose=1,
              callbacks=callbacks, validation_split=0.1, steps_per_epoch=x_train.shape[0]//batch_size)


