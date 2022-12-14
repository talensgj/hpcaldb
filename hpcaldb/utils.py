import os
import json

import numpy as np


def column_json_to_array(json_col):

    array_col = np.array([json.loads(row) for row in json_col])

    return array_col


def parse_filename(filename):
    """Function to strip the path and any extra extensions (.gz, .fz)."""

    # Strip the path.
    basename = os.path.basename(filename)

    if ('.fits' not in basename) and ('.txt' not in basename):
        raise ValueError(r"{filename} must be a FITS or TXT file.")

    # Check the extension.
    name, ext = os.path.splitext(basename)

    if ext in ['.gz', 'fz']:
        frame_name = name
        compression = ext
    else:
        frame_name = basename
        compression = None

    return frame_name, compression


def find_path_component(filename, index):

    # Find the full path and split it.
    filename = os.path.abspath(filename)
    splitpath = filename.strip(os.path.sep).split(os.path.sep)

    # Try to find the requested component.
    try:
        component = splitpath[index]
    except IndexError:
        component = None

    return component


def parse_workorder_filename(workorder):
    """ Extract information from the workorder filename.

    Parameters
    ----------
    workorder : str
        The workorder file to parse.

    Returns
    -------
    workdict : dict
        A dictionary of the information contained in the filename.

    """

    basename = os.path.splitext(os.path.basename(workorder))[0]

    workdict = dict()
    tmp = basename.split('_')
    if 'object' in basename:
        workdict['type'] = tmp[0]
        workdict['date'] = tmp[1]
        workdict['FNUM'] = int(tmp[2])
    else:
        workdict['type'] = tmp[0]
        workdict['date'] = tmp[1]
        workdict['part'] = int(tmp[2])
        workdict['IHUID'] = int(tmp[3])

    return workdict


def parse_imgsub_references_path(refs_path):
    """ Extract information from the references path.
    """

    refs_path = os.path.abspath(refs_path)

    # Split the path to get only the part realtive to the REF directory.
    # TODO having 'REF/' hardcoded is iffy.
    if 'REF/' in refs_path:
        refs_path = refs_path.split('REF/')[1]
    else:
        raise ValueError(f"The provided path {refs_path} does not match expectations.")

    refsdict = dict()
    tmp = refs_path.split('/')
    if tmp[0] == 'OBJECT_IHUID' and len(tmp) == 4:
        refsdict['type'] = tmp[0]
        refsdict['ihuid'] = int(tmp[1][3:])
        refsdict['object'] = tmp[2]
        refsdict['version'] = int(tmp[3][3:])
    elif tmp[0] == 'OBJECT_ONLY' and len(tmp) == 3:
        refsdict['type'] = tmp[0]
        refsdict['ihuid'] = None
        refsdict['object'] = tmp[1]
        refsdict['version'] = int(tmp[2][3:])
    else:
        raise ValueError(f"The provided path {refs_path} does not match expectations.")

    return refsdict, refs_path
