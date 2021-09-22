# coding: utf-8
import argparse
import datetime
import seaborn
import numpy as np
import tensorflow as tf
import recognition.util.image
import matplotlib.pyplot as plt
import recognition.util.path as path
import recognition.util.kmeans as kmeans

from recognition.parse.feature import Feature
from recognition.service.dataset import Dataset
from recognition.model.darknet.yolov3 import YoloV3


class Analyze:
    K_MEANS = 11
    PATH_TEMPORAL = 'temp'
    CURRENT_PALETTE = list(seaborn.xkcd_rgb.values())

    def __init__(self, filename):
        self._filename = filename
        self._classes = path.get_class_name()

        self._destination_path = path.resolve(Analyze.PATH_TEMPORAL)
        self._file_writer = tf.summary.create_file_writer(
            '{}/{}/analyze'.format(self._destination_path, datetime.datetime.today().strftime("%Y%m%d-%H%M%S")))

    def clustering_data(self, dataset):
        total = 0
        response = {}
        for example in dataset:
            _, _, extras = example
            for box in extras[Feature.LOCATION_BBOX]:
                clazz = box[4]
                key = next((key for key, lang in self._classes.items() if lang == clazz), None)
                response[key] = response[key] + 1 if key in response else 1
            total = total + 1
        return response, total

    def start(self):
        group = Feature(channel=Feature.CHANNEL)
        dataset = Dataset(filename=self._filename, feature=group, compression_type=Dataset.COMPRESSION_TYPE)
        dataset = dataset.load_and_map()

        # TODO: Incluir grilla de imágenes
        for step, example in enumerate(dataset):
            images, labels, extras = example
            with self._file_writer.as_default():
                images = tf.expand_dims(images, axis=0)
                tf.summary.image("image", images, step=step)

        data, total_data = self.clustering_data(dataset)
        data_figure = Analyze.plot_training_data(data, total_data)
        data_image = recognition.util.image.plot_to_image(data_figure)

        clustering = kmeans.clustering_data(dataset) / YoloV3.IMAGE_SIZE
        clustering_figure = Analyze.plot_clustering_data(clustering)
        clustering_image = recognition.util.image.plot_to_image(clustering_figure)

        clustering_info = Analyze.cluster_info(clustering)
        clustering_elbow_figure = Analyze.plot_clustering_elbow(clustering_info)
        clustering_elbow_image = recognition.util.image.plot_to_image(clustering_elbow_figure)

        with self._file_writer.as_default():
            tf.summary.image("Training data", data_image, step=0)
            tf.summary.image("Clusters", clustering_image, step=0)
            tf.summary.image("Elbow curve", clustering_elbow_image, step=0)

        count = 1
        figure = plt.figure(figsize=(15, 35))
        for k in range(2, Analyze.K_MEANS):
            clustering_objects = clustering_info[k]
            clusters = clustering_objects["clusters"]
            nearest_clusters = clustering_objects["nearest_clusters"]
            within_cluster_mean_dist = clustering_objects["within_cluster_mean_dist"]

            figure.add_subplot(Analyze.K_MEANS / 2, 2, count)
            Analyze.plot_clustering_result(plt, clusters, nearest_clusters, 1 - within_cluster_mean_dist, clustering)
            count = count + 1

        clusters_summary = recognition.util.image.plot_to_image(figure)
        with self._file_writer.as_default():
            tf.summary.image("Clusters summary", clusters_summary, step=0)

    @staticmethod
    def cluster_info(clustering):
        response = {}
        for k in range(2, 11):
            clusters, nearest_clusters, distances = kmeans.kmeans(
                clustering, k=k, dist=np.mean, seed=2)

            within_cluster_mean_dist = np.mean(
                distances[np.arange(distances.shape[0]), nearest_clusters])

            response[k] = {
                "clusters": clusters,
                "nearest_clusters": nearest_clusters,
                "distances": distances,
                "within_cluster_mean_dist": within_cluster_mean_dist
            }

        return response

    # noinspection PyShadowingNames
    @staticmethod
    def plot_clustering_result(plt, clusters, nearest_clusters, within_cluster_mean_dist, clustering):
        for cluster in np.unique(nearest_clusters):
            pick = nearest_clusters == cluster
            color = Analyze.CURRENT_PALETTE[cluster]

            plt.rc('font', size=8)
            plt.plot(clustering[pick, 0], clustering[pick, 1], "p",
                     color=color,
                     alpha=0.5, label="cluster = {}, N = {:6.0f}".format(cluster, np.sum(pick)))

            plt.text(clusters[cluster, 0],
                     clusters[cluster, 1], "c{}".format(cluster), fontsize=20, color="red")

            plt.title("Clusters")
            plt.xlabel("width")
            plt.ylabel("height")

        plt.legend(title="Mean IoU = {:5.4f}".format(within_cluster_mean_dist))

    @staticmethod
    def plot_clustering_data(objects):
        figure = plt.figure(figsize=(10, 10))
        plt.scatter(objects[:, 0], objects[:, 1], alpha=0.1)
        plt.title("Clusters", fontsize=20)
        plt.xlabel("normalized width", fontsize=20)
        plt.ylabel("normalized height", fontsize=20)
        return figure

    @staticmethod
    def plot_training_data(objects, total_images):
        keys = list(objects.keys())
        values = list(objects.values())
        y_position = np.arange(len(objects))

        figure = plt.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.barh(y_position, values)
        axes.set_yticks(y_position)
        axes.set_yticklabels(keys)
        axes.set_title("The total number of objects = {} in {} images".format(
            np.sum(values), total_images
        ))

        return figure

    @staticmethod
    def plot_clustering_elbow(objects):
        figure = plt.figure(figsize=(6, 6))
        plt.plot(np.arange(2, Analyze.K_MEANS),
                 [1 - objects[k]["within_cluster_mean_dist"] for k in range(2, Analyze.K_MEANS)], "o-")
        plt.title("within cluster mean of {}".format(np.mean))
        plt.ylabel("mean IOU")
        plt.xlabel("N clusters (= N anchor boxes)")
        return figure


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename',
                        type=str,
                        default=None,
                        metavar='',
                        help='')

    return parser.parse_args()


if __name__ == '__main__':
    arguments = process_args()

    analyze = Analyze(filename=arguments.filename)
    analyze.start()
