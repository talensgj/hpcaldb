from typing import Optional
import logging
from datetime import datetime, timedelta

from astropy.io import fits

from sqlalchemy import create_engine, select, and_, or_, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL
from sqlalchemy.sql.expression import text

from .models import Base, Workorder, Frame, TaskStatus, ProcessingQueue
from .models import MasterFrames, MasterBias, MasterDark, MasterFlat, MasterMask
from .models import FrameQuality, CalFrameQuality, SubFrameQuality
from .models import PSFStatistics, Astrometry, StarCatalog, AperPhotResult
from .models import ImgSubReferences

from . import utils

#############
#  LOGGING  #
#############

logger = logging.getLogger(__name__)

LOGDEBUG = logger.debug
LOGINFO = logger.info
LOGWARNING = logger.warning
LOGERROR = logger.error
LOGEXCEPTION = logger.exception

###############
# Basic setup #
###############

Session = sessionmaker()


def set_url(url, echo=False):
    """ Set the database URL.

    Parameters
    ----------
    url : str or dict or sqlalchemy.engine.URL
        Dictionary that can be passed to sqlalchemy.engine.URL, or a
        sqlalchemy.engine.URL see the sqlalchemy documentation.
    echo : bool
        Passed to sqlalchemy.create_engine, if True sqlalchemy generates
        verbose output (default is False).

    Returns
    -------
    engine : sqlalchemy.engine.Engine
        A sqlalchemy.engine.Engine instance.

    """

    LOGINFO("Connecting to the database.")

    if isinstance(url, dict):
        url = URL.create(drivername=url['drivername'],
                         username=url['username'],
                         password=url['password'],
                         host=url['host'],
                         port=url['port'],
                         database=url['database'])

    engine = create_engine(url, echo=echo)
    Session.configure(bind=engine)

    return engine


def create_all(engine):
    """ Create the calibration database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        The sqlalchemy.engine.Engine instance returned by set_url.

    """

    LOGINFO("Creating tables in the database.")

    Base.metadata.create_all(bind=engine)

    return


def drop_all(engine):
    """ Delete all tables from the database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        The sqlalchemy.engine.Engine instance returned by set_url.

    Notes
    -----
    This function should only be used in debugging environments.
    USE AT YOUR OWN RISK!

    """

    LOGINFO("Dropping all tables in the database.")

    Base.metadata.drop_all(bind=engine)

    return


def initpool(engine):
    """ Pass to mutiprocessing.Pool to ensure the database works when using
    multiprocessing.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        The sqlalchemy.engine.Engine instance returned by set_url.

    Notes
    -----
    Required to make sqlalchemy and multiprocessing play nice. Usage is as
    follows: multiprocessing.Pool(processes, hpcladb.initpool, (engine,))

    """

    # engine.dispose(close=False)  # TODO this is better
    engine.pool = engine.pool.recreate()

    return

###################################################
# Functions for interacting with the Core tables. #
###################################################


def insert_workorder(filename):
    """ Insert a workorder file into the database.

    Parameters
    ----------
    filename : str
        The name of the workorder file to insert.

    """

    LOGINFO(f"Inserting {filename} into the workorders table.")

    # Parse the filename.
    name, compression = utils.parse_filename(filename)

    # Connect to the database.
    with Session.begin() as session:

        # Check if the workorder already exists in the database.
        statement = select(Workorder).where(Workorder.name == name)
        workorder = session.execute(statement).scalar_one_or_none()

        if workorder is None:

            # Create the Workorder instance.
            workorder = Workorder(filename)

            # Add the workorder to the database.
            session.add(workorder)

        else:
            LOGWARNING(f"{filename} already in workorders table, skipping...")

    return


def find_workorder(filename):
    """ Find a specific workorder in the database.

    Parameters
    ----------
    filename : str
        The filename of the frame to search for.

    Returns
    -------
    workorder : hpcaldb.Workorder or None
        The database record for the workorder or None if it does not exist.

    """

    LOGINFO(f"Searching for {filename} in the workorders table.")

    name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(Workorder).where(Workorder.name == name)
        workorder = session.execute(statement).scalar_one_or_none()

    return workorder


def update_workorder_status(filename, completed=True):
    """ Update the status of a workorder in the database.

    Parameters
    ----------
    filename : str
        The filename of the frame to search for.
    completed : bool
        The new status (default: True).

    """

    LOGINFO(f"Updating 'completed' for {filename} in the workorders table.")

    name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(Workorder).where(Workorder.name == name)
        workorder = session.execute(statement).scalar_one_or_none()

        if workorder is not None:
            workorder.completed = completed
        else:
            LOGWARNING(f"Cannot update 'completed' for workorder {filename}, workorder not in database.")

        session.commit()

    return


def count_workorders():

    # Connect to the database.
    with Session() as session:

        # Build the query.
        statement = select(func.count(Workorder.name))
        statement = statement.where(Workorder.completed.is_(False))

        # Execute the query.
        count = session.execute(statement).scalar()

    return count


def insert_frames(filelist, hdr=None):
    """ Insert a (list of) new image frames into the database.

    Parameters
    ----------
    filelist : str or list[str]
        A frame or list of frames to be inserted in the frames table.
    hdr : astropy.io.fits.Header or None
        The header of the frame file, ignored if more than one file is provided
        (default is None).

    """

    # Check that input is list or string.
    if isinstance(filelist, str):
        filelist = [filelist]
    elif not isinstance(filelist, list):
        raise ValueError('filelist must be list or str')

    # Check that the hdr keyword is sane.
    if len(filelist) > 0 and hdr is not None:
        LOGWARNING(f"Received more than one frame to insert, ignoring the hdr keyword.")
        hdr = None

    # Connect to the database.
    with Session.begin() as session:

        # Loop over fits files.
        for filename in filelist:

            frame_name, compression = utils.parse_filename(filename)

            # Check if the frame already exists in the database.
            statement = select(Frame).where(Frame.frame_name == frame_name)
            frame = session.execute(statement).scalar_one_or_none()

            if frame is None:
                LOGINFO(f"Inserting frame {filename} into the frames table.")

                # Create the Frame instance and add the FrameQuality instance.
                frame = Frame(filename, hdr=hdr)
                frame.quality = FrameQuality(filename, hdr=hdr)

                # Add the frame to the database.
                session.add(frame)

            else:
                LOGWARNING(f"{filename} already in frames table, skipping...")

    return


def find_frame(filename):
    """ Find a specific frame in the database.

    Parameters
    ----------
    filename : str
        The filename of the frame to search for.

    Returns
    -------
    frame : hpcaldb.Frame or None
        The database record for the frame or None if it does not exist.

    """

    LOGINFO(f"Searching for {filename} in the frames table.")

    frame_name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

    return frame


def update_frame_compression(filename):
    """ Change the compression column for the specified FITS file.

    Parameters
    ----------
    filename : str
        The filename of the frame to update, the compression type is determined
        from the extension (only .fz and .gz are supported).

    """

    LOGINFO(f"Updating 'compression' for {filename} in the frames table.")

    frame_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

        if frame is not None:
            frame.compression = compression
        else:
            LOGWARNING(f"Cannot update 'compression' for frame {filename}, frame not in database.")

        session.commit()

    return


def count_frames(date_dir, ihuid=None):
    """ Count the number of frames in the database associated with a particular
        date.

    Parameters
    ----------
    date_dir : str
        The date_dir for which to count the frames.
    ihuid : int or None
        The IHUID for which to count the frames (default: None, count all IHUs).

    Returns
    -------
    count : int
        The number of frames in the database for this date (and IHUID).

    """

    # Connect to the database.
    with Session() as session:

        # Build the query.
        statement = select(func.count(Frame.FNUM))
        statement = statement.where(Frame.date_dir == date_dir)

        if ihuid is not None:
            statement = statement.where(Frame.IHUID == ihuid)

        # Execute the query.
        count = session.execute(statement).scalar()

    return count


##########################################################
# Functions for interacting with the Calibration tables. #
##########################################################


TASK_NAMES = ['calibrate', 'astrometry', 'aper_phot', 'magfit', 'imgsub', 'imgsub_phot']


def insert_into_processing_queue(filelist, task_name):
    """ Insert a task into the processing_queue table.

    Parameters
    ----------
    filelist : str or list[str]
        List of frames to insert into the processing queue.
    task_name : {'calibrate', 'astrometry', 'aper_phot', 'diff_phot'}
        The name of the task to be inserted into the processing queue.

    """

    # Check that input filelist is list or string.
    if isinstance(filelist, str):
        filelist = [filelist]
    elif not isinstance(filelist, list):
        raise ValueError('filelist must be list or str')

    # Check that the task name is valid.
    if task_name not in TASK_NAMES:
        raise ValueError(f'task_name = {task_name} not a valid task, must be one of {TASK_NAMES}')

    # Open a database session.
    with Session() as session:

        for filename in filelist:

            # Parse the filename.
            frame_name, compression = utils.parse_filename(filename)

            # Find the file in the frames table.
            statement = select(Frame).where(Frame.frame_name == frame_name)
            frame = session.execute(statement).scalar_one_or_none()

            LOGINFO(f"Trying to add task {task_name} to the processing_queue"
                    f" for frame {filename}.")

            if frame is None:
                LOGWARNING(f"Cannot add task {task_name} to the"
                           f" processing_queue table for frame {filename},"
                           f" frame not in database.")
                continue

            # Get the tasks and queued_tasks names.
            tasks = [item.task for item in frame.tasks]
            queued_tasks = [item.task for item in frame.queued_tasks]

            if task_name in queued_tasks:
                LOGWARNING(f"Cannot add task {task_name} to the"
                           f" processing_queue table for frame {filename},"
                           f" task already in queue.")
                continue

            if task_name not in tasks:
                frame.tasks.append(TaskStatus(task=task_name, status='pending'))
                frame.queued_tasks.append(ProcessingQueue(task=task_name))

            else:
                arg = tasks.index(task_name)
                if frame.tasks[arg].status == 'failure':
                    LOGWARNING(f"Task {task_name} already in the database for"
                               f" frame {filename} but status is 'failure',"
                               f" re-adding to the processing_queue table.")

                    frame.tasks[arg].status = 'pending'
                    frame.queued_tasks.append(ProcessingQueue(task=task_name))

                else:
                    LOGWARNING(f"Cannot add task {task_name} to the"
                               f" processing_queue table for frame {filename},"
                               f" task already completed.")
                    continue

        session.commit()

    return


def query_processing_queue(task_name,
                           include_ihu_ids=None,
                           exclude_ihu_ids=None,
                           wait_time_days=1,
                           max_tries=None,
                           force_process=False):
    """ Find all frames queued for a particular task.

    Parameters
    ----------
    task_name : str
        The name of the task to be queried, e.g. 'calibrate', 'astrometry', etc.
    include_ihu_ids : list[int] or None
        A list of the IHUIDs to be queried (default is None, implying all IHUIDs).
    exclude_ihu_ids : list[int] or None
        A list of the IHUIDs to be excluded (default is None).
    wait_time_days : float or None
        Time to wait after the task has been queried before returning its record
        again (default is 1 day, if None no wait is imposed).
    max_tries : int or None
        Maximum amount of tries a task can have (default is None, implying no limit).
    force_process : bool
        If True ignore wait_time_days and return all records regardless of
        the values of the max_tries and wait_time_days parameters (default is False).

    Returns
    -------
    frames : list
        List of hpcaldb.models.Frame objects.

    """

    # Check that the task name is valid.
    if task_name not in TASK_NAMES:
        raise ValueError(f'task_name = {task_name} not a valid task, must be one of {TASK_NAMES}')

    LOGINFO(f"Querying the processing_queue table for frames ready for the {task_name} task.")

    # Open a database session.
    with Session() as session:

        # Build the query.
        statement = select(ProcessingQueue)
        statement = statement.where(ProcessingQueue.task == task_name)

        if include_ihu_ids is not None:
            statement = statement.where(ProcessingQueue.IHUID.in_(include_ihu_ids))

        if exclude_ihu_ids is not None:
            statement = statement.where(ProcessingQueue.IHUID.not_in(exclude_ihu_ids))

        if force_process:
            # Set wait_time_days and max_tries to None to process all tasks.
            wait_time_days = None
            max_tries = None

        if wait_time_days is None:
            if max_tries is None:
                # Query all tasks.
                pass
            else:
                # Query all tasks with n_tries < max_tries.
                statement = statement.where(ProcessingQueue.n_tries < max_tries)
        else:
            wait_expired = func.date_sub(func.now(), text(f'interval {wait_time_days} day'))
            if max_tries is None:
                # Query tasks whose wait has expired and tasks with n_tries=0.
                statement = statement.where(or_(ProcessingQueue.time_updated < wait_expired,
                                                ProcessingQueue.n_tries == 0))
            else:
                # Query tasks whose wait has expired and where n_tries < max_tries and tasks with n_tries=0.
                statement = statement.where(or_(and_(ProcessingQueue.time_updated < wait_expired,
                                                     ProcessingQueue.n_tries < max_tries),
                                                ProcessingQueue.n_tries == 0))

        # Execute the query.
        result = session.execute(statement).scalars().all()

        # Update their n_tries parameter of the queried tasks.
        for queue in result:
            queue.n_tries += 1
        session.commit()

        # Get the list of Frame objects.
        frames = [item.frame for item in result]

    LOGINFO(f"Found {len(frames)} frames ready for the {task_name} task.")

    return frames


def remove_from_processing_queue(filelist, task_name, completed=True):
    """ Remove a task from the processing_queue table.

    Parameters
    ----------
    filelist : str or list[str]
        List of frames to remove from the processing queue.
    task_name : {'calibrate', 'astrometry', 'aper_phot', 'imgsub', 'imgsub_phot'}
        The name of the task to be removed from the processing queue.
    completed : bool
        If True, set the corresponding field in the frame_status table to
        'success' else set it to 'failure'.

    """

    # Check that input is list or string.
    if isinstance(filelist, str):
        filelist = [filelist]
    elif not isinstance(filelist, list):
        raise ValueError('filelist must be list or str')

    # Check that the task name is valid.
    if task_name not in TASK_NAMES:
        raise ValueError(f'task_name = {task_name} not a valid task, must be one of {TASK_NAMES}')

    with Session() as session:

        for filename in filelist:

            # Parse the filename.
            frame_name, compression = utils.parse_filename(filename)

            # Find the file in the frames table.
            statement = select(Frame).where(Frame.frame_name == frame_name)
            frame = session.execute(statement).scalar_one_or_none()

            LOGINFO(f"Trying to remove task {task_name} from the"
                    f" processing_queue for frame {filename}.")

            if frame is None:
                LOGWARNING(f"Cannot remove task {task_name} from the"
                           f" processing_queue table for frame {filename},"
                           f" frame not in database.")
                continue

            # Get the tasks and queued_tasks names.
            tasks = [item.task for item in frame.tasks]
            queued_tasks = [item.task for item in frame.queued_tasks]

            if task_name not in queued_tasks:
                LOGWARNING(f"Cannot remove task {task_name} from the"
                           f" processing_queue table for frame {filename},"
                           f" task not in queue.")
                continue

            # Update the task status.
            arg = tasks.index(task_name)
            if completed:
                frame.tasks[arg].status = 'success'
            else:
                frame.tasks[arg].status = 'failure'

            # Delete the task from the queue.
            arg = queued_tasks.index(task_name)
            del frame.queued_tasks[arg]

        session.commit()

    return


def count_tasks(ihuid):
    """ Count the number of pending tasks for a particular IHUID.

    Parameters
    ----------
    ihuid : int
        The IHUID for which to count the pending tasks.

    Returns
    -------
    task_counts : dict
        Dictionary with task_name, count as the key, value pairs.

    """

    task_counts = dict()

    # Connect to the database.
    with Session() as session:

        # Loop over the different tasks.
        for task_name in TASK_NAMES:

            # Build the query.
            statement = select(func.count(ProcessingQueue.task))
            statement = statement.where(ProcessingQueue.IHUID == ihuid, ProcessingQueue.task == task_name)

            # Execute the query.
            count = session.execute(statement).scalar()
            task_counts[task_name] = count

    return task_counts


def insert_masterbias(filename, calfiles, hdr=None):
    """ Insert a masterbias into the calibration database.

    Parameters
    ----------
    filename : str
        The filename of the masterbias.
    calfiles : list[str]
        The filenames of the individual bias frames used.
    hdr : astropy.io.fits.Header or None
        The header of the masterbias file (default is None).

    """

    LOGINFO(f"Inserting {filename} into the masterbias table.")

    # Parse the filename.
    bias_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        # Check if the masterbias frame already exists in the database.
        statement = select(MasterBias).where(MasterBias.bias_name == bias_name)
        masterbias = session.execute(statement).scalar_one_or_none()

        if masterbias is None:

            # Create the MasterBias instance.
            masterbias = MasterBias(filename, hdr=hdr)

            # Add the individual bias frames to the masterbias.
            for calframe in calfiles:
                frame_name, compression = utils.parse_filename(calframe)
                statement = select(Frame).where(Frame.frame_name == frame_name)
                frame = session.execute(statement).scalar_one_or_none()
                masterbias.bias_frames.append(frame)

            # Add the masterbias to the database.
            session.add(masterbias)

        else:
            LOGWARNING(f"{filename} already in masterbias table, skipping...")

    return


def find_masterbias(filename):
    """ Find a specific masterbias.

    Parameters
    ----------
    filename : str
        The filename of the masterbias to search for.

    Returns
    -------
    masterbias : hpcaldb.MasterBias or None
        The database record for the masterbias or None if it does not exist.

    """

    LOGINFO(f"Searching for {filename} in the masterbias table.")

    bias_name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(MasterBias).where(MasterBias.bias_name == bias_name)
        masterbias = session.execute(statement).scalar_one_or_none()

    return masterbias


def update_masterbias_compression(filename):
    """ Change the compression column for the specified FITS file.

    Parameters
    ----------
    filename : str
        The filename of the masterbias to update, the compression type is
        determined from the extension (only .fz and .gz are supported).

    """

    LOGINFO(f"Updating 'compression' for {filename} in the masterbias table.")

    bias_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        statement = select(MasterBias).where(MasterBias.bias_name == bias_name)
        masterbias = session.execute(statement).scalar_one_or_none()

        if masterbias is not None:
            masterbias.compression = compression
        else:
            LOGWARNING(f"Cannot update 'compression' for masterbias {filename}, file not in database.")

        session.commit()

    return


def query_masterbias(filename, hdr=None, max_day_delta=45):
    """ Query the calibration database for a suitable masterbias.

    Parameters
    ----------
    filename : str
        The filename of the FITS file to calibrate.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file to calibrate (default is None).
    max_day_delta : float
        The maximum age difference between the masterbias and the FITS file
        in days (default is 45 days).

    Returns
    -------
    masterbias : hpcaldb.MasterBias or None
        The database record of a suitable masterbias or None if one does not
        exist.

    """

    LOGINFO(f"Searching for a masterbias matching {filename}")

    # Read the header of the file.
    if hdr is None:
        hdr = fits.getheader(filename)

    # Time range from which to select the masterframe.
    dt = timedelta(days=max_day_delta)
    date = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d').date()
    tmin = date - dt
    tmax = date + dt

    # Search for appropriate masterbias files.
    with Session() as session:

        # Build the query.
        statement = select(MasterBias)
        statement = statement.where(MasterBias.CMID == hdr['CMID'],
                                    MasterBias.BIASVER == hdr['BIASVER'],
                                    MasterBias.RDMODE == hdr['RDMODE'],
                                    MasterBias.DATEOBS > tmin,
                                    MasterBias.DATEOBS < tmax,
                                    MasterBias.quality == 0)
        statement = statement.order_by(func.abs(func.datediff(MasterBias.DATEOBS, date)))

        # Execute the query.
        result = session.execute(statement).first()

    if result is not None:
        LOGINFO("Found a suitable masterbias.")
        masterbias = result[0]
    else:
        masterbias = None

    return masterbias


def insert_masterdark(filename, calfiles, hdr=None):
    """ Insert a masterdark into the calibration database.

    Parameters
    ----------
    filename : str
        The filename of the masterdark.
    calfiles : list[str]
        The filenames of the individual dark frames used.
    hdr : astropy.io.fits.Header or None
        The header of the masterdark file (default is None).

    """

    LOGINFO(f"Inserting {filename} into the masterdark table.")

    # Parse the filename.
    dark_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        # Check if the masterdark frame already exists in the database.
        statement = select(MasterDark).where(MasterDark.dark_name == dark_name)
        masterdark = session.execute(statement).scalar_one_or_none()

        if masterdark is None:

            # Create the MasterDark instance.
            masterdark = MasterDark(filename, hdr=hdr)

            # Add the individual dark frames to the masterdark.
            for calframe in calfiles:
                frame_name, compression = utils.parse_filename(calframe)
                statement = select(Frame).where(Frame.frame_name == frame_name)
                frame = session.execute(statement).scalar_one_or_none()
                masterdark.dark_frames.append(frame)

            # Add the masterdark to the database.
            session.add(masterdark)

        else:
            LOGWARNING(f"{filename} already in masterdark table, skipping...")

    return


def find_masterdark(filename):
    """ Find a specific masterdark.

    Parameters
    ----------
    filename : str
        The filename of the masterdark to search for.

    Returns
    -------
    masterdark : hpcaldb.MasterDark or None
        The database record for the masterdark or None if it does not exist.

    """

    LOGINFO(f"Searching for {filename} in the masterdark table.")

    dark_name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(MasterDark).where(MasterDark.dark_name == dark_name)
        masterdark = session.execute(statement).scalar_one_or_none()

    return masterdark


def update_masterdark_compression(filename):
    """ Change the compression column for the specified FITS file.

    Parameters
    ----------
    filename : str
        The filename of the masterdark to update, the compression type is
        determined from the extension (only .fz and .gz are supported).

    """

    LOGINFO(f"Updating 'compression' for {filename} in the masterdark table.")

    dark_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        statement = select(MasterDark).where(MasterDark.dark_name == dark_name)
        masterdark = session.execute(statement).scalar_one_or_none()

        if masterdark is not None:
            masterdark.compression = compression
        else:
            LOGWARNING(f"Cannot update 'compression' for masterdark {filename}, file not in database.")

        session.commit()

    return


def query_masterdark(filename, hdr=None, match_exptime=True,
                     max_day_delta=45, temp_tol=1, exp_tol=0.1):
    """ Query the calibration database for a suitable masterdark.

    Parameters
    ----------
    filename : str
        The filename of the FITS file to calibrate.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file to calibrate (default is None).
    match_exptime : bool
        If True match the exposure time of the masterdark and the FITS file,
        (default is True).
    max_day_delta : float
        The maximum age difference between the masterdark and the FITS file
        in days (default 45 days).
    temp_tol : float
        Maximum temperature difference between the masterdark and the FITS file
        in degrees (default is 1 degree).
    exp_tol : float
        Maximum exposure time difference between the masterdark and the FITS
        file in seconds (default is 0.1 seconds).

    Returns
    -------
    masterdark : hpcaldb.MasterDark or None
        The database record of a suitable masterdark or None if one does not
        exist.

    """

    LOGINFO(f"Searching for a masterdark matching {filename}")

    # Read the header of the file.
    if hdr is None:
        hdr = fits.getheader(filename)

    # Time range from which to select the masterframe.
    dt = timedelta(days=max_day_delta)
    date = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d').date()
    tmin = date - dt
    tmax = date + dt

    # Search for appropriate masterdark files.
    with Session() as session:

        # Build the query.
        statement = select(MasterDark)
        statement = statement.where(MasterDark.CMID == hdr['CMID'],
                                    MasterDark.DARKVER == hdr['DARKVER'],
                                    MasterDark.RDMODE == hdr['RDMODE'],
                                    MasterDark.DATEOBS > tmin,
                                    MasterDark.DATEOBS < tmax,
                                    func.abs(MasterDark.MEDNTEMP - hdr['CCDTEMP']) < temp_tol,
                                    MasterDark.quality == 0)

        if match_exptime:
            statement = statement.where(func.abs(MasterDark.EXPTIME - hdr['EXPTIME']) < exp_tol)
            statement = statement.order_by(func.abs(func.datediff(MasterDark.DATEOBS, date)))
        else:
            statement = statement.order_by(func.abs(MasterDark.EXPTIME - hdr['EXPTIME']),
                                           func.abs(func.datediff(MasterDark.DATEOBS, date)))

        # Execute the query.
        result = session.execute(statement).first()

    if result is not None:
        LOGINFO("Found a suitable masterdark.")
        masterdark = result[0]
    else:
        masterdark = None

    return masterdark


def insert_masterflat(filename, calfiles, hdr=None):
    """ Insert a masterflat into the calibration database.

    Parameters
    ----------
    filename : str
        The filename of the masterflat.
    calfiles : list[str]
        The filenames of the individual flat frames used.
    hdr : astropy.io.fits.Header or None
        The header of the masterflat file (default is None).

    """

    LOGINFO(f"Inserting {filename} into the masterflat table.")

    # Parse the filename.
    flat_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        # Check if the masterflat frame already exists in the database.
        statement = select(MasterFlat).where(MasterFlat.flat_name == flat_name)
        masterflat = session.execute(statement).scalar_one_or_none()

        if masterflat is None:

            # Create the MasterFlat instance.
            masterflat = MasterFlat(filename, hdr=hdr)

            # Add the individual flat frames to the masterflat.
            for calframe in calfiles:
                frame_name, compression = utils.parse_filename(calframe)
                statement = select(Frame).where(Frame.frame_name == frame_name)
                frame = session.execute(statement).scalar_one_or_none()
                masterflat.flat_frames.append(frame)

            # Add the masterflat to the database.
            session.add(masterflat)

        else:
            LOGWARNING(f"{filename} already in masterflat table, skipping...")

    return


def find_masterflat(filename):
    """ Find a specific masterflat.

    Parameters
    ----------
    filename : str
        The filename of the masterflat to search for.

    Returns
    -------
    masterflat : hpcaldb.MasterFlat or None
        The database record for the masterflat or None if it does not exist.

    """

    LOGINFO(f"Searching for {filename} in the masterflat table.")

    flat_name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(MasterFlat).where(MasterFlat.flat_name == flat_name)
        masterflat = session.execute(statement).scalar_one_or_none()

    return masterflat


def update_masterflat_compression(filename):
    """ Change the compression column for the specified FITS file.

    Parameters
    ----------
    filename : str
        The filename of the masterflat to update, the compression type is
        determined from the extension (only .fz and .gz are supported).

    """

    LOGINFO(f"Updating 'compression' for {filename} in the masterflat table.")

    flat_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        statement = select(MasterFlat).where(MasterFlat.flat_name == flat_name)
        masterflat = session.execute(statement).scalar_one_or_none()

        if masterflat is not None:
            masterflat.compression = compression
        else:
            LOGWARNING(f"Cannot update 'compression' for masterflat {filename}, file not in database.")

        session.commit()

    return


def query_masterflat(filename, hdr=None, max_day_delta=45):
    """ Query the calibration database for a suitable masterflat.

    Parameters
    ----------
    filename : str
        The filename of the FITS file to calibrate.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file to calibrate (default is None).
    max_day_delta : float
        The maximum age difference between the masterflat and the FITS file
        in days (default is 45 days).

    Returns
    -------
    masterflat : hpcaldb.MasterFlat or None
        The database record of a suitable masterflat or None if one does not
        exist.

    """

    LOGINFO(f"Searching for a masterflat matching {filename}")

    # Read the header of the file.
    if hdr is None:
        hdr = fits.getheader(filename)

    # Time range from which to select the masterframe.
    dt = timedelta(days=max_day_delta)
    date = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d').date()
    tmin = date - dt
    tmax = date + dt

    # Search for appropriate masterflat files.
    with Session() as session:

        # Build the query.
        statement = select(MasterFlat)
        statement = statement.where(MasterFlat.IHUID == hdr['IHUID'],
                                    MasterFlat.FLATVER == hdr['FLATVER'],
                                    MasterFlat.RDMODE == hdr['RDMODE'],
                                    MasterFlat.DATEOBS > tmin,
                                    MasterFlat.DATEOBS < tmax,
                                    MasterFlat.flat_type == 'high',
                                    MasterFlat.quality == 0)
        statement = statement.order_by(func.abs(func.datediff(MasterFlat.DATEOBS, date)))

        # Execute the query.
        result = session.execute(statement).first()

    if result is not None:
        LOGINFO("Found a suitable masterflat.")
        masterflat = result[0]
    else:
        masterflat = None

    return masterflat


def insert_mastermask(filename, low_masterflat, high_masterflat, hdr=None):
    """ Insert a mastermask in the calibration database.

    Parameters
    ----------
    filename : str
        The filename of the mastermask.
    low_masterflat : str
        The filename of the low masterflat used to generate the mastermask.
    high_masterflat : str
        The filename of the high masterflat used to generate the mastermask.
    hdr : astropy.io.fits.Header or None
        The header of the mastermask file (default is None).

    """

    LOGINFO(f"Inserting {filename} into the mastermask table.")

    # Parse the filename.
    mask_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        # Check if the mastermask frame already exists in the database.
        statement = select(MasterMask).where(MasterMask.mask_name == mask_name)
        mastermask = session.execute(statement).scalar_one_or_none()

        if mastermask is None:

            # Create the MasterMask instance.
            mastermask = MasterMask(filename, hdr=hdr)

            # Add the low and high masteflats to the mastermask.
            # TODO check that masterflat is found and has flat_type == 'low'?
            low_flat_name, compression = utils.parse_filename(low_masterflat)
            statement = select(MasterFlat).where(MasterFlat.flat_name == low_flat_name)
            low_masterflat = session.execute(statement).scalar_one_or_none()
            mastermask.low_masterflat = low_masterflat

            # TODO check that masterflat is found and has flat_type == 'high'?
            high_flat_name, compression = utils.parse_filename(high_masterflat)
            statement = select(MasterFlat).where(MasterFlat.flat_name == high_flat_name)
            high_masterflat = session.execute(statement).scalar_one_or_none()
            mastermask.high_masterflat = high_masterflat

            # Add the mastermask to the database.
            session.add(mastermask)

        else:
            LOGWARNING(f"{filename} already in mastermask table, skipping...")

    return


def find_mastermask(filename):
    """ Find a specific mastermask.

    Parameters
    ----------
    filename : str
        The filename of the mastermask to search for.

    Returns
    -------
    mastermask : hpcaldb.MasterMask or None
        The database record for the mastermask or None if it does not exist.

    """

    LOGINFO(f"Searching for {filename} in the mastermask table.")

    mask_name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(MasterMask).where(MasterMask.mask_name == mask_name)
        mastermask = session.execute(statement).scalar_one_or_none()

    return mastermask


def update_mastermask_compression(filename):
    """ Change the compression column for the specified FITS file.

    Parameters
    ----------
    filename : str
        The filename of the mastermask to update, the compression type is
        determined from the extension (only .fz and .gz are supported).

    """

    LOGINFO(f"Updating 'compression' for {filename} in the mastermask table.")

    mask_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        statement = select(MasterMask).where(MasterMask.mask_name == mask_name)
        mastermask = session.execute(statement).scalar_one_or_none()

        if mastermask is not None:
            mastermask.compression = compression
        else:
            LOGWARNING(f"Cannot update 'compression' for mastermask {filename}, file not in database.")

        session.commit()

    return


def query_mastermask(filename, hdr=None, max_day_delta=45):
    """ Query the calibration database for a suitable mastermask.

    Parameters
    ----------
    filename : str
        The filename of the FITS file to calibrate.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file to calibrate (default is None).
    max_day_delta : float
        The maximum age difference between the mastermask and the FITS file
        in days (default is 45 days).

    Returns
    -------
    mastermask : hpcaldb.MasterMask or None
        The database record of a suitable mastermask or None if one does not
        exist.

    """

    LOGINFO(f"Searching for a mastermask matching {filename}")

    # Read the header of the file.
    if hdr is None:
        hdr = fits.getheader(filename)

    # Time range from which to select the masterframe.
    dt = timedelta(days=max_day_delta)
    date = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d').date()
    tmin = date - dt
    tmax = date + dt

    # Search for appropriate masterflat files.
    with Session() as session:

        # Build the query.
        statement = select(MasterMask)
        statement = statement.where(MasterMask.CMID == hdr['CMID'],
                                    MasterMask.FLATVER == hdr['FLATVER'],
                                    MasterMask.RDMODE == hdr['RDMODE'],
                                    MasterMask.DATEOBS > tmin,
                                    MasterMask.DATEOBS < tmax,
                                    MasterMask.quality == 0)
        statement = statement.order_by(func.abs(func.datediff(MasterMask.DATEOBS, date)))

        # Execute the query.
        result = session.execute(statement).first()

    if result is not None:
        LOGINFO("Found a suitable mastermask.")
        mastermask = result[0]
    else:
        mastermask = None

    return mastermask


def insert_masterframes(filename, masterbias=None, masterdark=None,
                        masterflat=None, mastermask=None):
    """ Record which masterframes were used to calibrate a particular frame.

    Parameters
    ----------
    filename : str
        The filename of the FITS file that was calibrated.
    masterbias : hpcaldb.models.MasterBias or None
        The masterbias that was used to calibrate the FITS file (default is
        None).
    masterdark : hpcaldb.models.MasterDark or None
        The masterdark that was used to calibrate the FITS file (default is
        None).
    masterflat : hpcaldb.models.MasterFlat or None
        The masterflat that was used to calibrate the FITS file (default is
        None).
    mastermask : hpcaldb.models.MasterMask or None
        The mastermask that was used to calibrate the FITS file (default is
        None).

    """

    LOGINFO(f"Recording the masterframes used to calibrate {filename}")

    frame_name, compression = utils.parse_filename(filename)

    # Update the database with this information.
    with Session.begin() as session:

        # Find the frame for which we are adding masters.
        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

        # Add the masterframes.
        if frame.masterframes is None:
            frame.masterframes = MasterFrames()

        # Set the values for the various masters.
        # TODO this overwrites existing values, is that OK?
        frame.masterframes.masterbias = masterbias
        frame.masterframes.masterdark = masterdark
        frame.masterframes.masterflat = masterflat
        frame.masterframes.mastermask = mastermask

        # Commit changes.
        session.commit()

    return


def insert_calframe_metrics(filename, fits_info, mask_counts, hdr=None):
    """ Record the quality metrics of a calibrated frame in the database.

    Parameters
    ----------
    filename : str
        The filename of the calibrated FITS file.
    fits_info : dict
        Result dictionary of pipetrex.imageutils.quality.frame_source_info
    mask_counts : dict
        Result dictionary of pipetrex.ficalib.calutils.fiinfo_maskinfo
    hdr : astropy.io.fits.Header or None
        The header of the FITS file (default is None).

    """

    LOGINFO(f"Recording the quality metrics of calibrated frame {filename}")

    frame_name, compression = utils.parse_filename(filename)

    with Session() as session:

        # Find the frame for which we are adding the quality metrics.
        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

        # Add the quality metrics.
        if frame.calframe_quality is None:
            frame.calframe_quality = CalFrameQuality(filename, fits_info, mask_counts, hdr=hdr)
        else:
            # TODO what if there already are quality metrics?
            LOGWARNING(f"{filename} already has quality metrics, skipping...")

        # Add the quality metrics to the database.
        session.commit()

    return


def insert_subframe_metrics(filename, fits_info, mask_counts, hdr=None):
    """ Record the quality metrics of a subtracted frame in the database.

    Parameters
    ----------
    filename : str
        The filename of the subtracted FITS file.
    fits_info : dict
        Result dictionary of pipetrex.imageutils.quality.frame_source_info
    mask_counts : dict
        Result dictionary of pipetrex.ficalib.calutils.fiinfo_maskinfo
    hdr : astropy.io.fits.Header or None
        The header of the FITS file (default is None).

    """

    LOGINFO(f"Recording the quality metrics of subtracted frame {filename}")

    frame_name, compression = utils.parse_filename(filename)

    with Session() as session:

        # Find the frame for which we are adding the quality metrics.
        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

        # Add the quality metrics.
        if frame.subframe_quality is None:
            frame.subframe_quality = SubFrameQuality(filename, fits_info, mask_counts, hdr=hdr)
        else:
            # TODO what if there already are quality metrics?
            LOGWARNING(f"{filename} already has quality metrics, skipping...")

        # Add the quality metrics to the database.
        session.commit()

    return


def insert_psf_statistics(filename, psfstats, hdr=None):
    """ Record the PSF statistics of a frame in the database.

    Parameters
    ----------
    filename : str
        The filename of the FITS file for which the astrometry was solved.
    psfstats : dict
        Result dictionary of pipetrex.photometry.srcextract.get_psf_statistics
    hdr : astropy.io.fits.Header or None
        The header of the FITS file (default is None).

    """

    LOGINFO(f"Recording the PSF statistics of {filename}")

    frame_name, compression = utils.parse_filename(filename)

    with Session() as session:

        # Find the frame for which we are adding the quality metrics.
        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

        # Add the PSF statistics.
        if frame.psf is None:
            frame.psf = PSFStatistics(filename, psfstats, hdr=hdr)
        else:
            # TODO what if there already are PSF statistics?
            LOGWARNING(f"{filename} already has PSF statistics, skipping...")

        # Add the PSF statistics to the database.
        session.commit()

    return


def insert_astrometry(filename, program, astromres, hdr=None):
    """ Record the astrometric solution of a frame in the database.

    Parameters
    ----------
    filename : str
        The filename of the FITS file for which the astrometry was solved.
    program : str
        The program used to obtain this astrometric solution.
    astromres : piastrom.piastromres.AstromRes or pipetrex.astrometry.hatpi_refine.AstromRes
        The AstromRes result class instance.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file (default is None).

    """

    LOGINFO(f"Recording the astrometric solution of {filename}")

    frame_name, compression = utils.parse_filename(filename)

    with Session() as session:

        # Find the frame for which we are adding the astrometry.
        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

        # Add the astrometry.
        if frame.astrometry is None:
            frame.astrometry = Astrometry(filename, program, astromres, hdr=hdr)
        else:
            # TODO what if there already is astrometry?
            LOGWARNING(f"{filename} already has an astrometric solution, skipping...")

        # Add the astrometry to the database.
        session.commit()

    return


def insert_star_catalog(filename, hdr=None):
    """ Insert star catalogs into the database.

    Parameters
    ----------
    filename : str
        The filename of the FITS star catalog.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file (default is None).

    """

    LOGINFO(f"Inserting {filename} in the star catalogs table.")

    catalog_name, compression = utils.parse_filename(filename)
    catalog_dir = utils.find_path_component(filename, -2)

    with Session.begin() as session:

        # Check if the catalog already exists in the database.
        statement = select(StarCatalog).where(StarCatalog.catalog_name == catalog_name.endswith,
                                              StarCatalog.catalog_dir == catalog_dir)
        starcat = session.execute(statement).scalar_one_or_none()

        if starcat is None:

            # Create the StarCatalog instance.
            starcat = StarCatalog(filename, hdr=hdr)

            # Add the catalog to the database.
            session.add(starcat)

        else:
            LOGWARNING(f"{filename} already in the star catalog table, skipping...")

    return


def find_star_catalog(filename):
    """ Find a specific star catalog in the database.

    Parameters
    ----------
    filename : str
        The filename of the star catalog to search for.

    Returns
    -------
    starcat : hpcaldb.StarCatalog or None
        The database record for the catalog or None if it does not exist.

    """

    LOGINFO(f"Searching for {filename} in the star catalogs table.")

    catalog_name, compression = utils.parse_filename(filename)
    catalog_dir = utils.find_path_component(filename, -2)

    with Session() as session:

        statement = select(StarCatalog).where(StarCatalog.catalog_name == catalog_name,
                                              StarCatalog.catalog_dir == catalog_dir)
        starcat = session.execute(statement).scalar_one_or_none()

    return starcat


def update_star_catalog_compression(filename):
    """ Change the compression column for the specified FITS file.

    Parameters
    ----------
    filename : str
        The filename of the star catalog to update, the compression type is
        determined from the extension (only .fz and .gz are supported).

    """

    LOGINFO(f"Updating 'compression' for {filename} in the star_catalogs table.")

    catalog_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        statement = select(StarCatalog).where(StarCatalog.catalog_name == catalog_name)
        starcat = session.execute(statement).scalar_one_or_none()

        if starcat is not None:
            starcat.compression = compression
        else:
            LOGWARNING(f"Cannot update 'compression' for star catalog {filename}, file not in database.")

        session.commit()

    return


def query_star_catalog(filename, hdr=None, max_dist=1.):
    # TODO is the query optimal? Most recent might not be best, use closest epoch instead?
    """ Query the calibration database for a suitable star catalog.

    Parameters
    ----------
    filename : str
        The filename of the FITS file to calibrate.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file to calibrate (default is None).
    max_dist : float
        The maximum allowable distance between the telescope and catalog
        pointing in degrees.

    Returns
    -------
    starcat : hpcaldb.StarCatalog or None
        The database record of a suitable star catalog or None if one does not
        exist.

    """

    LOGINFO(f"Searching for a star catalog matching {filename}")

    # Read the header of the file.
    if hdr is None:
        hdr = fits.getheader(filename)

    # Search for an appropriate star catalog file.
    with Session() as session:

        # Build the query.
        statement = select(StarCatalog)
        statement = statement.where(StarCatalog.OBJECT == hdr['OBJECT'],
                                    StarCatalog.distance(hdr['CRVAL1'], hdr['CRVAL2']) < max_dist)
        statement = statement.order_by(StarCatalog.time_updated.desc())  # TODO is this most recent first?

        # Execute the query.
        result = session.execute(statement).first()

        # If no suitable starcat found try searching by position only.
        if result is None:

            LOGWARNING(f"Initial query failed, searching by position only.")

            # Build the query.
            statement = select(StarCatalog)
            statement = statement.where(StarCatalog.distance(hdr['CRVAL1'], hdr['CRVAL2']) < max_dist)
            statement = statement.order_by(StarCatalog.time_updated.desc())  # TODO is this most recent first?

            # Execute the query.
            result = session.execute(statement).first()

    if result is not None:
        LOGINFO("Found a suitable star catalog.")
        starcat = result[0]
    else:
        starcat = None

    return starcat


def insert_aper_phot_result(filename, catalog, hdr=None):
    """ Insert an aperture photometry file in the database.

    Parameters
    ----------
    filename : str
        The filename of the FITS photometry file.
    catalog : hpcald.models.StarCatalog
        The star catalog that was used to extract the photometry.
    hdr : astropy.io.fits.Header or None
        The header of the FITS file (default is None).

    """

    LOGINFO(f"Inserting {filename} in the aper_phot_results table.")

    file_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        # Check of the file already exists in the database.
        statement = select(AperPhotResult).where(AperPhotResult.file_name == file_name)
        aperphot = session.execute(statement).scalar_one_or_none()

        if aperphot is None:

            # Create the AperPhotResult instance.
            aperphot = AperPhotResult(filename, hdr=hdr)

            # Add the catalog.
            aperphot.catalog = catalog

            # Add the aperture photometry to the database.
            session.add(aperphot)

        else:
            LOGWARNING(f"{filename} already in the aper_phot_results table, skipping...")

    return


def find_aper_phot_result(filename):
    """ Find a specific aperture photometry file in the database.

    Parameters
    ----------
    filename : str
        The filename of the aperture photometry file to search for.

    Returns
    -------
    aperphot : hpcaldb.AperPhotResult or None
        The database record for the aperture photometry file ot None if it does
        not exist.

    """

    LOGINFO(f"Searching for {filename} in the aper_phot_results table.")

    file_name, compression = utils.parse_filename(filename)

    with Session() as session:

        statement = select(AperPhotResult).where(AperPhotResult.file_name == file_name)
        aperphot = session.execute(statement).scalar_one_or_none()

    return aperphot


def update_aper_phot_compression(filename):
    """ Change the compression column for the specified FITS file.

    Parameters
    ----------
    filename : str
        The filename of the photometry file to update, the compression type is
        determined from the extension (only .fz and .gz are supported).

    """

    LOGINFO(f"Updating 'compression' for {filename} in the aper_phot_results table.")

    file_name, compression = utils.parse_filename(filename)

    with Session.begin() as session:

        statement = select(AperPhotResult).where(AperPhotResult.file_name == file_name)
        aperphot = session.execute(statement).scalar_one_or_none()

        if aperphot is not None:
            aperphot.compression = compression
        else:
            LOGWARNING(f"Cannot update 'compression' for photometry file {filename}, file not in database.")

        session.commit()

    return


def insert_imgsub_references(refs_path: str,
                             astroref_frame: str,
                             photoref_frames: list[str]
                             ) -> None:
    """ Insert a set of image subtraction reference files into the database.

    Parameters
    ----------
    refs_path: str
        Full path to the set of reference files.
    astroref_frame : str
        The calibrated frame used as the astrometric reference.
    photoref_frames : list[str]
        The calibrated frames used to create the photometric reference.

    """

    _, refs_dir = utils.parse_imgsub_references_path(refs_path)

    LOGINFO(f"Inserting the image subtraction reference files in {refs_dir} into the imgsub_references table.")

    with Session.begin() as session:

        # Check if the reference files alread exist in the database.
        statement = select(ImgSubReferences).where(ImgSubReferences.refs_dir == refs_dir)
        imgsubrefs = session.execute(statement).scalar_one_or_none()

        if imgsubrefs is None:

            # Create the ImgSubReferences instance.
            imgsubrefs = ImgSubReferences(refs_path)

            # Add the object frames used as the astrometric reference.
            frame_name, _ = utils.parse_filename(astroref_frame)
            statement = select(Frame).where(Frame.frame_name == frame_name)
            frame = session.execute(statement).scalar_one_or_none()
            imgsubrefs.astroref_frame = frame

            # Add the individual object frames to the photometric reference.
            for photoref_frame in photoref_frames:
                frame_name, _ = utils.parse_filename(photoref_frame)
                statement = select(Frame).where(Frame.frame_name == frame_name)
                frame = session.execute(statement).scalar_one_or_none()
                imgsubrefs.photoref_frames.append(frame)

            # Add the reference files to the database.
            session.add(imgsubrefs)

        else:
            LOGWARNING(f"{refs_dir} already in the 'imgsub_references' table, skipping...")

    return


def find_imgsub_references(refs_path: str) -> Optional[ImgSubReferences]:
    """ Find a set of image subtraction references in the database.

    Parameters
    ----------
    refs_path: str
        Full path to the set of reference files.

    Returns
    -------
    imgsubrefs : hpcaldb.ImgSubReferences or None
        The database record of the image subtraction references or None
        if it does not exist.

    """

    _, refs_dir = utils.parse_imgsub_references_path(refs_path)

    LOGINFO(f"Searching for {refs_dir} in the 'imgsub_references' table.")

    with Session() as session:

        statement = select(ImgSubReferences).where(ImgSubReferences.refs_dir == refs_dir)
        imgsubrefs = session.execute(statement).scalar_one_or_none()

    return imgsubrefs


def query_imgsub_references(filename: str,
                            hdr: Optional[fits.Header] = None,
                            match_ihuid: bool = True
                            ) -> Optional[ImgSubReferences]:
    """ Query the database for suitable image subtraction references.

    Parameters
    ----------
    filename : str
        The filename of the FITS file that will be image subtracted.
    hdr: astropy.io.Header or None
        The header of the FITS file that will be image subtracted
        (default is None).
    match_ihuid : bool
        If True the IHUID key will be matched, instead of just the OBJECT key.

    Returns
    -------
    imgsubrefs : hpcaldb.ImgSubReferences or None
        The database record of suitable image subtraction references or None
        if no references exist.

    """

    LOGINFO(f"Searching for image subtraction references matching {filename}")

    # Read the header of the file.
    if hdr is None:
        hdr = fits.getheader(filename)

    # Search for appropriate image subtraction references.
    with Session() as session:

        # Build the query.
        statement = select(ImgSubReferences)
        statement = statement.where(ImgSubReferences.OBJECT == hdr['OBJECT'],
                                    ImgSubReferences.quality == 0)

        if match_ihuid:
            statement = statement.where(ImgSubReferences.IHUID == hdr['IHUID'],
                                        ImgSubReferences.refs_type == 'OBJECT_IHUID')
        else:
            statement = statement.where(ImgSubReferences.refs_type == 'OBJECT_ONLY')

        statement = statement.order_by(ImgSubReferences.refs_version.desc())

        # Execute the query.
        result = session.execute(statement).first()

    if result is not None:
        LOGINFO("Found suitable image subtraction references.")
        imgsubrefs = result[0]
    else:
        imgsubrefs = None

    return imgsubrefs


def insert_imgsub_references_used(filename, imgsubrefs):
    """ Record which references were used to perform image subtraction on a
        frame.

    Parameters
    ----------
    filename : str
        The filename of the FITS file that was image subtracted.
    imgsubrefs : hpcaldb.ImgSubReferences
        The image subtraction references used.

    """

    LOGINFO(f"Recording the references used to subtract {filename}")

    frame_name, compression = utils.parse_filename(filename)

    # Update the database with this information.
    with Session.begin() as session:

        # Find the frame for which we are adding references.
        statement = select(Frame).where(Frame.frame_name == frame_name)
        frame = session.execute(statement).scalar_one_or_none()

        # Set the imgsubrefs.
        frame.imgsubrefs = imgsubrefs

        # Commit changes.
        session.commit()

    return


def query_status(num_days=1):

    ref_time = func.date_sub(func.now(), text(f'interval {num_days} day'))

    workdict = dict()
    taskdict = dict()
    with Session() as session:

        # Count incomplete workorders.
        statement = select(func.count(Workorder.name))
        statement = statement.where(Workorder.completed == False)  # noqa
        count = session.execute(statement).scalar()
        workdict['incomplete'] = count

        # Count newly inserted workorders.
        statement = select(func.count(Workorder.name))
        statement = statement.where(Workorder.time_created > ref_time)
        count = session.execute(statement).scalar()
        workdict['inserted'] = count

        # Count newly completed workorders.
        statement = select(func.count(Workorder.name))
        statement = statement.where(Workorder.time_updated > ref_time)
        statement = statement.where(Workorder.completed == True)  # noqa
        count = session.execute(statement).scalar()
        workdict['completed'] = count

        # Count status of tasks.
        for task in TASK_NAMES:

            taskdict[task] = dict()

            # Count all pending tasks.
            statement = select(func.count(TaskStatus.FNUM))
            statement = statement.where(TaskStatus.task == task)
            statement = statement.where(TaskStatus.status == 'pending')
            count = session.execute(statement).scalar()
            taskdict[task]['pending'] = count

            # Count newly failed tasks.
            statement = select(func.count(TaskStatus.FNUM))
            statement = statement.where(TaskStatus.task == task)
            statement = statement.where(TaskStatus.status == 'failure')
            statement = statement.where(TaskStatus.time_updated > ref_time)
            count = session.execute(statement).scalar()
            taskdict[task]['failure'] = count

            # Count newly completed tasks.
            statement = select(func.count(TaskStatus.FNUM))
            statement = statement.where(TaskStatus.task == task)
            statement = statement.where(TaskStatus.status == 'succcess')
            statement = statement.where(TaskStatus.time_updated > ref_time)
            count = session.execute(statement).scalar()
            taskdict[task]['success'] = count

    return workdict, taskdict


def main():

    return


if __name__ == '__main__':
    main()
