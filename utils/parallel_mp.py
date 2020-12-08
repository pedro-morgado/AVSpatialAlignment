import multiprocessing as mp


class ParallelMP:
    def __init__(self, n_workers):
        import multiprocessing as mp
        self.n_workers = n_workers
        self.q_job = mp.Queue(maxsize=n_workers*5)
        self.q_ret = mp.Queue(maxsize=n_workers*5)

    def scheduler(self, jobs):
        for i, job in enumerate(jobs):
            self.q_job.put((i, job))
        for _ in range(self.n_workers):
            self.q_job.put((None, None))

    def worker(self, fcn):
        while True:
            i, job = self.q_job.get()
            if i is None:
                break
            if i % 100 == 0:
                print(f"{i}/{self.n_jobs}")
            ret = fcn(*job)
            if ret:
                self.q_ret.put((i, ret))
        self.q_ret.put((None, None))

    def gather(self):
        done = 0
        outputs = [None] * self.n_jobs
        while done != self.n_workers:
            i, ret = self.q_ret.get()
            if i is None:
                done += 1
                continue
            outputs[i] = ret
        return outputs

    def apply(self, fcn, arg_list):
        self.n_jobs = len(arg_list)
        scheduler_worker = mp.Process(target=self.scheduler, args=(arg_list, ), daemon=True)
        scheduler_worker.start()

        workers = []
        for _ in range(self.n_workers):
            w = mp.Process(target=self.worker, args=(fcn, ), daemon=True)
            w.start()
            workers += [w]

        outputs = self.gather()

        scheduler_worker.join()
        for w in workers:
            w.join()

        return outputs
