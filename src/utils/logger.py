import numpy as np
import pandas as pd


class CommRecord:
    def __init__(self, from_party_id, to_party_id, size):
        """
        Record of communication size

        Parameters
        ----------
        from_party_id : int
            ID of the party that sends data
        to_party_id : int
            ID of the party that receives data
        size : int
            size of data
        """
        self.from_party_id = from_party_id
        self.to_party_id = to_party_id
        self.size = size

    def __str__(self):
        return f"{self.from_party_id},{self.to_party_id},{self.size}"


class CommLogger:
    def __init__(self, n_parties, path=None):
        """
        Logger of communication size

        Parameters
        ----------
        n_parties : int
            number of parties. A server is also included as a party.
        """
        self.n_parties = n_parties
        self.path = path
        self.comm_records = []

        self.in_comm = [0. for _ in range(n_parties)]   # communication size received by each party
        self.out_comm = [0. for _ in range(n_parties)]  # communication size sent by each party

    def comm(self, from_party_id, to_party_id, size: int):
        """
        Record communication size in bytes

        Parameters
        ----------
        from_party_id : int
            ID of the party that sends data
        to_party_id : int
            ID of the party that receives data
        size : int
            size of data
        """
        self.comm_records.append((from_party_id, to_party_id, size))
        self.in_comm[to_party_id] += size
        self.out_comm[from_party_id] += size

    def broadcast(self, from_party_id, size):
        """
        Record the communication from a party to all other parties

        Parameters
        ----------
        from_party_id : int
            ID of the party that sends data
        size : int
            size of data
        """
        for to_party_id in range(self.n_parties):
            if to_party_id != from_party_id:
                self.comm(from_party_id, to_party_id, size)

    def receive_all(self, to_party_id, size):
        """
        Record the communication from all other parties to a party

        Parameters
        ----------
        to_party_id : int
            ID of the party that receives data
        size : int
            size of data
        """
        for from_party_id in range(self.n_parties):
            if from_party_id != to_party_id:
                self.comm(from_party_id, to_party_id, size)

    def save_log(self):
        """
        Save the log to a csv file
        """
        columns = ["From", "To", "Size"]
        df = pd.DataFrame(self.comm_records, columns=columns)
        df.to_csv(self.path, index=False)

    @classmethod
    def load_log(cls, path):
        """
        Save the log to a csv file

        Parameters
        ----------
        path : str
            path of the csv file
        """
        data = pd.read_csv(path)
        n_parties = max(data["From"].max(), data["To"].max()) + 1
        logger = cls(n_parties)
        comm_records = data.values

        def add_row(row):
            logger.in_comm[row[1]] += row[2]
            logger.out_comm[row[0]] += row[2]
        np.apply_along_axis(add_row, axis=1, arr=comm_records)
        logger.comm_records = comm_records.tolist()

        return logger

    @property
    def total_comm_bytes(self):
        # each communication is counted twice (one in from_party_id, one in to_party_id)
        return (sum(self.in_comm) + sum(self.out_comm)) / 2

    @property
    def max_in_comm_bytes(self):
        return max(self.in_comm)

    @property
    def max_out_comm_bytes(self):
        return max(self.out_comm)

    @property
    def total_comm_kB(self):
        return self.total_comm_bytes / 1024

    @property
    def max_in_comm_kB(self):
        return self.max_in_comm_bytes / 1024

    @property
    def max_out_comm_kB(self):
        return self.max_out_comm_bytes / 1024

    @property
    def total_comm_MB(self):
        return self.total_comm_kB / 1024

    @property
    def max_in_comm_MB(self):
        return self.max_in_comm_kB / 1024

    @property
    def max_out_comm_MB(self):
        return self.max_out_comm_kB / 1024

    @property
    def total_comm_GB(self):
        return self.total_comm_MB / 1024

    @property
    def max_in_comm_GB(self):
        return self.max_in_comm_MB / 1024

    @property
    def max_out_comm_GB(self):
        return self.max_out_comm_MB / 1024


