import numpy as np
import pandas as pd
import sqlite3
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
from typing import Callable
from data import transform
__all__ = ["HiggsGenerator", "HiggsDataset"]

# dataset statistics
DATASIZE = 11_000_000
HEADERS = ["label",
           "lepton_pT", "lepton_eta", "lepton_phi",
           "missing_energy_magnitude", "missing_energy_phi",
           "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b-tag",
           "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b-tag",
           "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b-tag",
           "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b-tag",
           "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb",]
N_NOISE = 5170877
N_SIGNAL = 5829123


class HiggsGenerator(IterableDataset):
    def __init__(self,
                 db_path, 
                 transform: Callable = transform.Identity(),
                 seed: int = None,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.db_path = db_path
        self.transform = transform
        self.table_name = "higgs"
        self.connection = None
        self.cursor = None

    def _get_connection(self):
        # This ensures each worker process gets its own unique connection
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
        return self.cursor

    def __iter__(self):
        return self

    def __next__(self):
        cursor = self._get_connection()
        query = f"SELECT label, jet_1_phi, jet_2_phi, jet_3_phi, jet_4_phi \
                  FROM {self.table_name} \
                  WHERE (id = ? AND label = ?) OR (id = ? AND label = ?)"
        i = self.rng.integers(N_NOISE).item()
        j = self.rng.integers(N_SIGNAL).item()
        cursor.execute(query, (i,0,j,1))
        rows = cursor.fetchall()
        c = int(rows[0][0])
        nul = np.array(rows[c][1:])
        alt = np.array(rows[1-c][1:])
        return self.transform((nul, alt))



class HiggsDataset(Dataset):
    def __init__(self,
                 db_path, 
                 transform: Callable = transform.Identity(),
                 seed: int = 0,):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.db_path = db_path
        self.transform = transform
        self.order_nul = self.rng.permutation(N_NOISE).tolist()
        self.order_alt = self.rng.permutation(N_SIGNAL).tolist()
        self.table_name = "higgs"
        self.connection = None
        self.cursor = None

    def __len__(self):
        return min(N_NOISE, N_SIGNAL)

    def _get_connection(self):
        # This ensures each worker process gets its own unique connection
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
        return self.cursor

    def __getitem__(self, idx):
        cursor = self._get_connection()
        query = f"SELECT label, jet_1_phi, jet_2_phi, jet_3_phi, jet_4_phi \
                  FROM {self.table_name} \
                  WHERE (id = ? AND label = ?) OR (id = ? AND label = ?)"
        i = self.order_nul[idx]
        j = self.order_alt[idx]
        cursor.execute(query, (i,0,j,1))
        rows = cursor.fetchall()
        c = int(rows[0][0])
        nul = np.array(rows[c][1:])
        alt = np.array(rows[1-c][1:])
        return self.transform((nul, alt))

    def collate_hook(self, any):
        self.order_nul = self.rng.permutation(N_NOISE).tolist()
        self.order_alt = self.rng.permutation(N_SIGNAL).tolist()
        return any



def csv_to_sql(csv_path, db_path, table_name = "higgs"):
    r"""Writes the given csv data into a SQL database."""
    conn = sqlite3.connect(db_path)
    m = n = 0
    CHUNKSIZE = 100_000
    pbar = tqdm(total=DATASIZE // CHUNKSIZE)
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE, header=None, names=HEADERS):
        # add separate id for null and alt samples
        chunk.insert(0, "id", None)
        nul = chunk["label"]==0
        alt = chunk["label"]==1
        chunk.loc[nul, "id"] = range(m, m+nul.sum())
        chunk.loc[alt, "id"] = range(n, n+alt.sum())
        m += nul.sum()
        n += alt.sum()
        chunk.to_sql(table_name, conn, if_exists='append', index=False)
        pbar.update()
        pbar.set_description(f"processing...")
    pbar.close()

    print("creating index...")
    conn.execute(f'CREATE INDEX idx_{table_name}_id_label ON {table_name}(id, label)')
    conn.close()



def access_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"PRAGMA index_list({table_name})")
    indices = cur.fetchall()
    for index in indices:
        print(f"{index=}")
        index_name = index[1]
        cur.execute(f"PRAGMA index_info({index_name})")
        cols = cur.fetchall()
        print(f"Index: {index_name}")
        for col in cols:
            print(f"\t{col}")

    # cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE label = ?", (1,))
    # row = cur.fetchone()
    # print(f"{row=}")



def main():
    csv_path = Path("data/cache/higgs/HIGGS.csv")
    db_path = csv_path.with_suffix(".db")
    csv_to_sql(csv_path, db_path)
    # access_db(db_path, table_name="higgs")

if __name__ == '__main__':
    main()


