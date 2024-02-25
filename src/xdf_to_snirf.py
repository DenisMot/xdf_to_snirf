import numpy as np
import matplotlib.pyplot as plt
import pyxdf
import os
import datetime
from dateutil import tz
import mne
from mne_nirs.io import write_raw_snirf


def get_NIRS_and_Event_streams(x_file):
    """
    Load the xdf file and returns only the NIRS and Event streams
    """
    # load only the NIRS and Event streams
    data, header = pyxdf.load_xdf(
        filename=x_file,
        select_streams=[{"type": "NIRS"}, {"type": "Event"}],
        synchronize_clocks=True,
        # NOTE: dejitter is necessary to get closer to the oxy4 data
        dejitter_timestamps=True,
        verbose=False,
    )

    # find the nirs stream among the list of streams
    for i in range(len(data)):
        if data[i]["info"]["type"][0] == "NIRS":
            nirsStream = data[i]
            break
    # find the Event stream among the list of streams
    for i in range(len(data)):
        if data[i]["info"]["type"][0] == "Event":
            eventStream = data[i]
            break

    return nirsStream, eventStream


def print_xdf_stream_labels(stream):
    """
    Print the labels of the channels by channel number
    """

    channels = []
    for chan in stream["info"]["desc"][0]["channels"][0]["channel"]:
        label = chan["label"]
        unit = chan["unit"]
        type = chan["type"]
        channels.append({"label": label, "unit": unit, "type": type})
    print("Found {} channels: ".format(len(channels)))
    for i in range(len(channels)):
        print(
            "  {:02d}: {} ({} {})".format(
                i,
                channels[i]["label"][0][8:],  # remove the first 8 characters
                channels[i]["type"][0],
                channels[i]["unit"][0],
            )
        )


def print_xdf_stream_labels_and_first_last_data(stream):
    """
    Print the labels of the channels + first data value by channel number
    """
    channels = []
    for chan in stream["info"]["desc"][0]["channels"][0]["channel"]:
        label = chan["label"]
        unit = chan["unit"]
        type = chan["type"]
        channels.append({"label": label, "unit": unit, "type": type})
    print("Found {} channels: ".format(len(channels)))
    for i in range(len(channels)):
        print(
            "  {:02d}: {} ({} {}) [{:5.3f}...{:5.3f}]".format(
                i,
                channels[i]["label"][0][8:],  # remove the first 8 characters
                channels[i]["type"][0],
                channels[i]["unit"][0],
                stream["time_series"][0, i],
                stream["time_series"][-1, i],
            )
        )


def xdf_reorganize_channels_as_in_snirf(nirsStream):
    """
    Reorganize the xdf stream channels and info/desc0/channels0/ as it is in the snirf file
    """
    # if the stream already has 16 channels, do nothing
    if len(nirsStream["info"]["desc"][0]["channels"][0]["channel"]) == 16:
        print("Stream already has 16 channels")
        return nirsStream

    # # modify the data according to the ARTINIS matlab code
    # # data.dataTimeSeries = 1./exp(log(10).* [rawvals(:, 2:2:end) rawvals(:, 1:2:end)]);%change dataTimeSeries to correct values
    # In the XDF file, we have 34 channels, but only 16 are of interest
    # only channels 0 to 7 and 24 to 31 are effectively used
    # and the order should be changed to match the snirf file (small wavelength first)

    new_order = [
        1,
        3,
        5,
        7,
        25,
        27,
        29,
        31,
        0,
        2,
        4,
        6,
        24,
        26,
        28,
        30,
    ]

    # keep only the 16 channels used and in the snirf order
    # NOTE: we do this for the time series AND the channel labels in info/desc0/channels0/
    channels = []
    time_series = np.zeros((len(nirsStream["time_series"]), len(new_order)))
    for i in range(len(new_order)):
        iNew = new_order[i]
        channels.append(nirsStream["info"]["desc"][0]["channels"][0]["channel"][iNew])
        time_series[:, i] = nirsStream["time_series"][:, iNew]

    # modify the stream itself
    nirsStream["info"]["desc"][0]["channels"][0]["channel"] = channels
    nirsStream["time_series"] = time_series

    # convert the modified stream to the correct values for snirf
    # NOTE: comment out => only change the order of the channels (for verification)
    # NOTE: the following two lines are equivalent
    # nirsStream["time_series"] = 1.0 / 10.0 ** nirsStream["time_series"]
    nirsStream["time_series"] = 1.0 / np.exp(np.log(10) * nirsStream["time_series"])

    # convert the names of the channel to the snirf format
    for i in range(len(channels)):
        label = nirsStream["info"]["desc"][0]["channels"][0]["channel"][i]["label"][0]
        label_str = label[8:]  # remove the
        tk = label_str.split(" ")
        S = int(tk[2][1:])
        D = int(tk[0][2:])
        W = int(tk[-1][1:-3])
        # sources labels (by half the same)
        if i < len(channels) / 2:
            S = int(S / 2)
        if i >= len(channels) / 2:
            S = int((S + 1) / 2)
        new_label = "S{}_D{} {}".format(S, D, W)
        nirsStream["info"]["desc"][0]["channels"][0]["channel"][i]["label"][
            0
        ] = new_label


def make_xdf_time_relative_to_first_data(nirsStream, eventStream):
    """
    Make the time of the NIRS and Event streams relative to the first data time
    As this is in the snirf file
    """

    nirs_time = nirsStream["time_stamps"]
    event_time = eventStream["time_stamps"]

    # make time relative to the beginning of the recording
    t_zero = nirs_time[0]
    nirs_time = nirs_time - t_zero
    event_time = event_time - t_zero

    nirsStream["time_stamps"] = nirs_time
    eventStream["time_stamps"] = event_time


# Montage for the OXY4
dig_points_ReArm_NeuArm = {
    "nz": [0.400, 85.900, -47.600],
    "a1": [83.900, -16.600, -56.700],
    "a2": [-83.800, -18.600, -57.200],
    "cz": [-0.461, -8.416, 101.365],
    "iz": [0.200, -120.500, -25.800],
    "S1": [15.122, 12.567, 92.751],
    "S2": [55.348, 12.215, 69.807],
    "S3": [58.686, -30.779, 80.338],
    "S4": [18.350, -31.372, 101.863],
    "S5": [-18.221, 14.421, 91.198],
    "S6": [-57.118, 11.411, 68.420],
    "S7": [-60.350, -33.460, 78.860],
    "S8": [-19.193, -31.499, 101.587],
    "D1": [38.717, -8.599, 90.282],
    "D2": [-39.835, -9.543, 89.911],
}

# divide all the coordinates by 1000 to get them in m
for key in dig_points_ReArm_NeuArm:
    # standard unit are in m
    dig_points_ReArm_NeuArm[key] = [x / 1000 for x in dig_points_ReArm_NeuArm[key]]
    # standard head is smaller than the one that was recorded
    # the 1.08 factor is to have the inter-SD distance of 0.03 m
    dig_points_ReArm_NeuArm[key] = [x / 1.08 for x in dig_points_ReArm_NeuArm[key]]


def distance(S1, D1):
    S1 = np.array(S1[:])
    D1 = np.array(D1[:])
    distance = np.sqrt(np.sum((D1 - S1) ** 2))
    return distance


def check_source_detector_distance(dig_points_ReArm_NeuArm):
    source_detector_distance = []
    for k in ["S1", "S2", "S3", "S4"]:
        D = dig_points_ReArm_NeuArm["D1"]
        S = dig_points_ReArm_NeuArm[k]
        source_detector_distance.append(distance(D, S))
    for k in ["S5", "S6", "S7", "S8"]:
        D = dig_points_ReArm_NeuArm["D2"]
        S = dig_points_ReArm_NeuArm[k]
        source_detector_distance.append(distance(D, S))

    # create a figure with two subplots
    plt.figure(figsize=(10, 5))
    # first subplot on the left half : boxplot of the source-detector distance
    ax = plt.subplot(1, 2, 1)
    ax.boxplot(source_detector_distance)
    ax.set_title("Source-Detector distance")
    ax.set_ylabel("Distance (m)")
    # second subplot with a scatter plot of the source-detector distance
    ax = plt.subplot(1, 2, 2)
    ax.scatter(range(len(source_detector_distance)), source_detector_distance)
    ax.set_ylabel("Distance (m)")
    plt.show()


def set_dig_montage(channels):
    # set the channel locations
    montage = mne.channels.make_dig_montage(
        ch_pos=channels,
        nasion=channels["nz"],
        lpa=channels["a1"],
        rpa=channels["a2"],
        hsp=None,
        hpi=None,
        coord_frame="head",
    )
    return montage


def set_rearm_dig_montage():
    return set_dig_montage(dig_points_ReArm_NeuArm)


def get_date_and_patientID_from_xdf_file_name(xdf_fullFile):
    """
    Get the date and patient ID from the xdf file name
    """
    # get the file name without the path and extension from the xdf file
    file_name = os.path.basename(xdf_fullFile)
    tokens = file_name.split("_")
    if len(tokens[1]) == 8:  # date
        patientID = tokens[0]
        date = tokens[1]
    elif len(tokens[2]) == 8:  # date
        patientID = tokens[0] + "_" + tokens[1]
        date = tokens[2]
    else:
        patientID = "Unknown"
        date = "Unknown"

    Y = date[0:4]
    m = date[4:6]
    d = date[6:8]

    Y = int(Y)
    m = int(m)
    d = int(d)

    H = 12
    M = 0
    S = 0

    TZ = tz.gettz("Europe/Paris")

    date = datetime.datetime(Y, m, d, H, M, S, 0, tzinfo=TZ)

    return date, patientID


def set_date_and_patientID(raw, xdf_fullFile):
    """
    Set the date and patient ID in the raw mne object from the xdf file name
    """
    date, subjectID = get_date_and_patientID_from_xdf_file_name(xdf_fullFile)
    raw.info["subject_info"] = {"first_name": subjectID}
    raw.set_meas_date(date.timestamp())


def set_results_path():
    # if the pwd is "notebooks", then go up one level to the root directory
    if os.path.basename(os.getcwd()) == "notebooks":
        # create or use the results directory
        new_fpath = os.path.join("..", "results")
        if not os.path.exists(new_fpath):
            os.makedirs(new_fpath)
    elif os.path.basename(os.getcwd()) == "xdf_to_snirf":
        new_fpath = os.path.join("results")
    return new_fpath


def create_snirf_in_results(xdf_fullFile, nirsStream, eventStream):
    """
    Create a snirf file named as the xdf file in the results directory
    """

    # get the file name without the path and extension from the xdf file
    file_name = os.path.basename(xdf_fullFile)
    new_fname = os.path.splitext(file_name)[0] + ".snirf"

    new_fpath = set_results_path()
    new_file_name = os.path.join(new_fpath, new_fname)

    n_channels = nirsStream["time_series"].shape[1]
    sampling_freq = nirsStream["info"]["nominal_srate"][0]
    ch_names = [
        n["label"][0] for n in nirsStream["info"]["desc"][0]["channels"][0]["channel"]
    ]

    nirs_info = mne.create_info(
        ch_names=ch_names,
        ch_types="fnirs_cw_amplitude",
        sfreq=sampling_freq,
        verbose=None,
    )

    # add the wavelength in nirs_info["chs"][0]["loc"][9]
    for i in range(n_channels):
        nirs_info["chs"][i]["loc"][9] = int(ch_names[i][-3:])

    nirs_info["description"] = "XDF to SNIRF"

    nirs_data = nirsStream["time_series"].T

    raw = mne.io.RawArray(
        data=nirs_data,
        info=nirs_info,
        verbose=None,
    )

    event_data = eventStream["time_series"]
    event_time = eventStream["time_stamps"]

    for i in range(len(event_data)):
        description = event_data[i][0][3:]
        duration = 5.0
        onset = event_time[i]
        raw.annotations.append(onset, duration, description)

    set_date_and_patientID(raw, xdf_fullFile)

    raw.set_montage(set_rearm_dig_montage())

    write_raw_snirf(raw, new_file_name, add_montage=True)

    return new_file_name


def xdf2snirf(xdf_fullFile):

    # Suppress the INFO log messages
    import logging

    logger = logging.getLogger("mne")
    logger.setLevel(logging.WARNING)

    nirsStream, eventStream = get_NIRS_and_Event_streams(xdf_fullFile)
    xdf_reorganize_channels_as_in_snirf(nirsStream)
    make_xdf_time_relative_to_first_data(nirsStream, eventStream)
    new_fname = create_snirf_in_results(xdf_fullFile, nirsStream, eventStream)

    new = mne.io.read_raw_snirf(new_fname, preload=True, verbose="CRITICAL")
    print(f"Created {new_fname}")
    print(
        f"  {new.info['nchan']} channels, {new.n_times} time points at {new.info['sfreq']} Hz) "
    )
    print(" ")

    return new_fname


# if run as a script

if __name__ == "__main__":
    # define the files to be tested
    xdf_fullFile = "../data/reference/015_AgePie_20211112_1_r(1).xdf"
    xdf_fullFile = "../data/reference/C1P07_20210802_1_r.xdf"

    def get_xdf_files_in_directory(directory, return_abspath=False):
        """
        Get all *.xdf files in the directory and return their relative paths.
        If return_abspath is True, return the absolute paths.
        """
        extension = ".xdf"

        print("cwd = ", os.getcwd())
        print("directory = ", os.path.abspath(directory)    )

        # get the files in the directory
        files = [f for f in os.listdir(directory) if f.endswith(extension)]
        full_files = [os.path.join(directory, f) for f in files]
        if return_abspath:
            full_files = [os.path.abspath(f) for f in full_files]
        full_files = [os.path.normpath(f) for f in full_files]
        full_files.sort()  # always easier like this
        return full_files

    # get all files in the directory
    xdf_files = get_xdf_files_in_directory("data/reference", return_abspath=True)
    print(
        len(xdf_files), "xdf files: [\n", xdf_files[0], "\n... \n", xdf_files[-1], "]"
    )

    # process all the xdf files
    for file in xdf_files:
        xdf_fullFile = os.path.abspath(os.path.join(os.getcwd(), file))
        xdf2snirf(xdf_fullFile)
