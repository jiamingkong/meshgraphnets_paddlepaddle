class DataLoader(object):
    def __init__(self, dataset, batch_size=1, num_workers=0, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        if self.batch_size == 1:
            self.collate_fn = lambda x: x[0]

    def __iter__(self):
        dataset_iterator = iter(self.dataset)
        end_of_dataset = False
        while not end_of_dataset:
            batch_data = []
            for i in range(self.batch_size):
                try:
                    data = next(dataset_iterator)
                    batch_data.append(data)
                except StopIteration:
                    end_of_dataset = True
                    # give out the last batch
                    break

            if len(batch_data) > 0:
                yield self.collate_fn(batch_data)
