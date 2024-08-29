import os
import pooch
import pytest
import tempfile


"""Data for testing and examples are hosted on Google Drive. Pooch is used to download
this data by parsing registry.csv and prefetching these data once per testing session.
Downloads are deleted when tests are completed."""


"""Parse the filename, sha256, and hyperlinks for hosted files"""


def registry():
    table = os.path.join(os.path.dirname(os.path.realpath(__file__)), "registry.csv")
    hashes = {}
    urls = {}
    with open(table, "r") as f:
        for line in f:
            f, h, u = line.split(",")
            hashes[f] = h
            urls[f] = u
    return hashes, urls


"""Modify Pooch to for prefetch of hosted files"""


class PrefetchPooch(pooch.Pooch):
    def prefetch(self, files=None):
        files = self.registry.keys() if not files else files
        for file in files:
            self._assert_file_in_registry(file)
            self.fetch(
                fname=file,
                processor=(
                    pooch.Unzip(extract_dir="./")
                    if os.path.splitext(file)[1] == ".zip"
                    else None
                ),
            )


"""This fixture prefetches hosted data to a temporary folder that is deleted on 
completion"""


@pytest.fixture(scope="session")
def data(path=None, files=None):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        hashes, urls = registry()
        data = PrefetchPooch(
            path=tmp,
            base_url="{version}/",
            registry=hashes,
            urls=urls,
        )
        data.prefetch(files)
        yield data
