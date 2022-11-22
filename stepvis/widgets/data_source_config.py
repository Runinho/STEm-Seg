# widget allowing to change the data source
import json
from pathlib import Path
from typing import Union

from PyQt5.QtCore import pyqtSlot as Slot, pyqtSignal as Signal
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QComboBox, QLineEdit, QFormLayout, \
    QGroupBox, QPushButton
from stepvis.inference_loader import OfflineInferenceDataProvider, InferenceSplitType, \
    OnlineInferenceDataProvider, InferenceDataProvider, InferenceDataProviderSequence
from stemseg.config import config

class ConfManager:
    """registers setter and getter for easier loading and saving of configs"""
    def __init__(self):
        self.fields = []

    def register_field(self, name, getter, setter):
        self.fields.append((name, getter, setter))

    def register_sub_conf(self, name, sub_conf):
        self.fields.append((name, sub_conf.get, sub_conf.set))

    def get_sub(self, name):
        conf = ConfManager()
        self.register_sub_conf(name, conf)
        return conf

    def get(self):
        # call getter
        return {name: getter() for name, getter, _ in self.fields}

    def set(self, obj):
        # call setter
        for name, _, setter in self.fields:
            if name in obj:
                setter(obj[name])
            else:
                print(f"could not find {name} in loaded conf")

class CachedConfManager(ConfManager):
    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def load(self):
        if self.filename.exists():
            obj = json.load(open(self.filename, "r"))
            self.set(obj)
            print(f"loaded conf from {self.filename}: {obj}")
            return obj
        else:
            print(f"could not load cached config from {self.filename}")

    def save(self):
        obj = self.get()
        json.dump(obj, open(self.filename, "w"))
        print(f"config saved to {self.filename}")

class BaseDataSourceConfigWidget(QWidget):
    def __init__(self, parent=None, conf_manager=ConfManager()):
        super().__init__(parent)

        # split type
        self.source_type = QComboBox()
        self.source_type.addItems(["Validation", "Training"])
        self.source_type.currentTextChanged.connect(self.split_changed)

        # TODO: do we still need this?
        self.split = "Validation"

        # setup config manager
        self.cm = conf_manager
        self.cm.register_field("source_type",
                               self.source_type.currentText,
                               self.source_type.setCurrentText)

    @Slot()
    def split_changed(self, new_split):
        self.split = new_split

    def get(self):
        return {"split": self.source_type.currentText()}

class OfflineDataSourceConfigWidget(BaseDataSourceConfigWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        # path
        self.folder = QLineEdit()
        self.folder.setPlaceholderText("")

        location_group = QGroupBox(self.tr("load predictions from file"))
        group_layout = QFormLayout()
        group_layout.addRow(self.tr("data split"), self.source_type)
        group_layout.addRow(self.tr("prediction directory"), self.folder)
        location_group.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(location_group)
        self.setLayout(layout)

        # setup config manager
        self.cm.register_field("pred_dir", self.folder.text, self.folder.setText)

    def get(self):
        obj = super().get()
        obj["pred_dir"] = self.folder.text()
        return obj

class OnlineDataSourceConfigWidget(BaseDataSourceConfigWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        # model weight location
        self.file = QLineEdit()
        self.file.setPlaceholderText("")

        location_group = QGroupBox(self.tr("generate predictions with model"))
        group_layout = QFormLayout()
        group_layout.addRow(self.tr("data split"), self.source_type)
        group_layout.addRow(self.tr("model checkpoint"), self.file)
        location_group.setLayout(group_layout)

        layout = QVBoxLayout()
        layout.addWidget(location_group)
        self.setLayout(layout)

        # setup config manager
        self.cm.register_field("model_file", self.file.text, self.file.setText)

    def get(self):
        obj = super().get()
        obj["checkpoint"] = self.file.text()

    def set(self, obj):
        super().set(obj)
        if "checkpoint" in obj:
            self.file.setText(obj["checkpoint"])

class DataSourceConfigWidget(QWidget):

    def __init__(self, parent=None, sequence_changed_cb=None):
        super().__init__(parent)
        self.sequence_changed_cb = sequence_changed_cb
        self.data_provider: Union[None, InferenceDataProvider] = None

        # helper to get and set the config values
        self.cm = CachedConfManager(Path("vis_data_source_cache.json"))

        # Offline or Online Data Source?
        self.source_type = QComboBox()
        self.source_type.addItems(["Online", "Offline"])
        self.source_type.currentTextChanged.connect(self.source_type_changed)

        # Offline
        # self.model_config =
        self.offline = \
            OfflineDataSourceConfigWidget(self,
                                          conf_manager=self.cm.get_sub("offline"))
        self.offline.hide()
        self.online = \
            OnlineDataSourceConfigWidget(self,
                                         conf_manager=self.cm.get_sub("online"))

        # action to load the sequence obj
        self.load_sequences = QPushButton(self.tr("load source"))
        self.load_sequences.clicked.connect(self.load_available_sequences)
        self.sequence_status = QLabel("no sequence information available")

        self.sequences = QComboBox()
        self.sequences.currentTextChanged.connect(self.sequence_changed)

        # register values in conf_manager
        self.cm.register_field("source_type",
                               self.source_type.currentText,
                               self.source_type.setCurrentText)
        self.cm.register_field("sequence",
                               self.sequences.currentText,
                               self.sequences.setCurrentText)

        # Create Layout
        self.vlayout = QFormLayout()
        self.vlayout.addRow("Data source", self.source_type)
        self.vlayout.addWidget(self.online)
        self.vlayout.addWidget(self.offline)
        self.vlayout.addWidget(self.load_sequences)
        self.vlayout.addWidget(self.sequence_status)
        self.vlayout.addRow("Sequence", self.sequences)
        self.setLayout(self.vlayout)

        # load from file
        conf = self.cm.load()
        self.load_available_sequences()
        self.cm.set(conf)

    def source_type_changed(self, new_type):
        print(f"data source changed to {new_type}")

        self.online.hide()
        self.offline.hide()

        # display correct data source TAB
        if new_type == "Online":
            self.online.show()
        elif new_type == "Offline":
            self.offline.show()
        else:
            print(f"Error: Could not select unkown source type: {new_type}")

        self.cm.save()

    @Slot()
    def load_available_sequences(self):

        config.load_global("kitti_step_2.yaml")

        inference_split_mapping = {"Validation": InferenceSplitType.VAL,
                                   "Training": InferenceSplitType.TRAIN}

        source_type = self.source_type.currentText()
        if source_type == "Online":
            model_path = self.online.file.text()
            if len(model_path) == 0:
                self.sequence_status.setText("model_file is empty")
                return
            split = inference_split_mapping[self.online.source_type.currentText()]
            self.data_provider = OnlineInferenceDataProvider(split_type=split,
                                                             model_path=model_path)
        elif source_type == "Offline":
            folder = self.offline.folder.text()
            if len(folder) == 0:
                self.sequence_status.setText("prediction folder name is empty")
                return
            split = inference_split_mapping[self.offline.source_type.currentText()]
            self.data_provider = OfflineInferenceDataProvider(split_type=split,
                                                              pred_location=self.offline.folder.text())
        else:
            print(f"error loading data (unkown source type: {source_type})")

        self.data_provider.init_providers()

        # fill combo box
        ids = self.data_provider.get_sequence_ids()
        old_id = self.sequences.currentText()
        self.sequences.clear()
        self.sequences.addItems(ids)

        # update current sequence
        if old_id in ids:
            self.sequences.setCurrentText(old_id)
        else:
            self.sequences.setCurrentText(ids[0])
        self.sequence_status.setText(f"found {len(ids)} sequences. "
                                     f"in {json.dumps(self.cm.get()[source_type.lower()])}.")

    def sequence_changed(self, new_sequence):
        data_provider = self.data_provider.get_sequence_by_id(new_sequence)
        # self.data_provider_changed
        if data_provider is not None:
            self.cm.save()
            self.sequence_changed_cb(data_provider)
        else:
            self.sequence_status.setText(f"failed to load sequence `{new_sequence}`")
