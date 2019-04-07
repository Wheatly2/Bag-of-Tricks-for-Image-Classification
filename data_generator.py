import os
import tensorflow as tf


class Dataset(object):

    def __init__(self, data_dir, batch_size, crop_size, means=[114.0,123.0,125.0],
            num_readers=1, shuffle=False, repeat=False):
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.num_readers = num_readers
        self.shuffle = shuffle
        self.repeat = repeat
        self.means = means

        self.tfrecords_list = [os.path.join(data_dir, i)  
           for i in os.listdir(data_dir) if i.endswith('tfrecord')]
    
    
    def _preprocess_image(self, image, label):
        
        image = tf.to_float(image)
        # mean substraction
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        for i in range(3):
            channels[i] -= self.means[i]
        distort_image = tf.concat(axis=2, values=channels)
        # flip
        distort_image = tf.image.random_flip_left_right(distort_image)
        # pad
        pad = 4
        distort_image = tf.pad(distort_image, [[pad, pad], [pad, pad], [0, 0]])
        # crop
        distort_image = tf.random_crop(distort_image, 
                                         [self.crop_size, self.crop_size, 3])
        
        return image



    def _parse_function(self, example_proto):

        features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/channels': tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/label': tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_bmp(parsed_features['image/encoded'])
        label = parsed_features['image/label']
        self._preprocess_image(image, label)
        sample = {'image': image, 'label': label}
        return sample
    

    def get_one_shot_iterator(self):
        
        dataset = (tf.data.TFRecordDataset(self.tfrecords_list, num_parallel_reads=self.num_readers)
                   .map(self._parse_function, num_parallel_calls=self.num_readers))
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.repeat:
            dataset = dataset.repeat()  
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        
        return dataset.make_one_shot_iterator()
