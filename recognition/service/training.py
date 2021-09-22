# coding: utf-8
import time
import datetime
import tensorflow as tf
import recognition.util.path as path

from recognition.model.model import Model


class Training:
    """Se encarga de iniciar el entrenamiento del modelo"""

    MAX_TO_KEEP = 3
    MAX_OUTPUTS = 9
    CHECKPOINT_STEP = 3
    PATH_TEMPORAL = 'temp'

    def __init__(self, model, **kwargs):
        if not isinstance(model, Model):
            raise ValueError('the model are not an instance of recognition.model.Model')

        strategy = kwargs.get('strategy', None)
        learning_rate = kwargs.get('learning_rate', 0.005)
        destination_path = kwargs.get('destination_path', None)

        self._model = model
        self._strategy = strategy
        self._learning_rate = learning_rate
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)

        if destination_path is None:
            destination_path = path.resolve(Training.PATH_TEMPORAL)
        self._destination_path = destination_path

        self._checkpoint = None
        self._checkpoint_manager = None
        self._save_checkpoint_step = Training.CHECKPOINT_STEP

        self._logdir = '{}/{}/{}'.format(
            self._destination_path, datetime.datetime.today().strftime("%Y%m%d-%H%M%S"), self._model.name)
        self._file_writer = tf.summary.create_file_writer(self._logdir)

        self._summary_accuracy = tf.keras.metrics.Sum(name="accuracy", dtype=tf.float32)

    def checkpoint_manager(self, save_checkpoint_step=None, max_to_keep=None):
        """Configurar el guardado de los puntos de control"""
        if save_checkpoint_step is None:
            save_checkpoint_step = Training.CHECKPOINT_STEP

        if max_to_keep is None:
            max_to_keep = Training.MAX_TO_KEEP

        self._save_checkpoint_step = save_checkpoint_step
        self._checkpoint = tf.train.Checkpoint(model=self._model, optimizer=self._optimizer)
        self._checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint, directory='{}/checkpoint'.format(self._destination_path), max_to_keep=max_to_keep)
        self._checkpoint.restore(self._checkpoint_manager.latest_checkpoint)

    @tf.function
    def train_step(self, images, labels):
        """Inicia el paso de predicción del modelo ya sea en modo sencillo o usando una estrategia de distribución"""
        if self._strategy is None:
            return self.step_fn(images, labels)

        loss_value = self._strategy.experimental_run_v2(self.step_fn, args=(images, labels))
        return self._strategy.reduce(tf.distribute.ReduceOp.SUM, loss_value, axis=None)

    def step_fn(self, images, labels):
        """Realiza la predicción del modelo"""
        with tf.GradientTape() as tape:
            predictions = self._model(images, training=True)
            loss_value = self.loss_step(labels, predictions)
            accuracy_value = self.accuracy_step(labels, predictions)

        gradient = tape.gradient(loss_value, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradient, self._model.trainable_variables))

        self._summary_accuracy.update_state(accuracy_value)
        return loss_value

    def loss_step(self, labels, predictions):
        """Inicia el paso para obtener la pérdida generada por el modelo"""
        raise NotImplementedError

    def accuracy_step(self, labels, predictions):
        """Inicia el paso para obtener la precisión generada por el modelo"""
        raise NotImplementedError

    def start(self, dataset, max_outputs=None):
        """Recorre el set de datos para iniciar el entrenamiento"""
        if dataset is None:
            raise ValueError('The dataset is none')

        if max_outputs is None:
            max_outputs = Training.MAX_OUTPUTS

        if self._checkpoint_manager is None:
            print('Warning: the checkpoint manager is none')

        tf.summary.trace_on(graph=True, profiler=True)
        for step, example in enumerate(dataset):
            start = time.time()
            images, labels = example
            with self._file_writer.as_default():
                tf.summary.image("image", images, step=step, max_outputs=max_outputs)

            loss_value = self.train_step(images=images, labels=labels)
            if self._checkpoint_manager is not None:
                if step > 0 and (step % self._save_checkpoint_step) == 0:
                    print('Save checkpoint {}'.format(step))
                    self._checkpoint_manager.save()

            print('Time taken for 1 epoch {} sec'.format(time.time() - start))
            with self._file_writer.as_default():
                if step == 0:
                    tf.summary.trace_export('trace', step=step, profiler_outdir=self._logdir)

                tf.summary.scalar('loss', loss_value, step=step)
                tf.summary.scalar('accuracy', self._summary_accuracy.result(), step=step)

                tf.summary.scalar('learning rate', data=self._learning_rate, step=step)
                for weight in self._model.trainable_weights:
                    tf.summary.histogram(weight.name, weight, step=step)

            print('Step {}, Loss: {}, Accuracy: {}\n'.format(step, loss_value, self._summary_accuracy.result()))
            self._summary_accuracy.reset_states()
            self._file_writer.flush()

        self._model.save_weights('model/{}.h5'.format(self._model.name))
