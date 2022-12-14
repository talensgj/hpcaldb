import os
import glob
import logging
from typing import Optional, Any
from datetime import datetime

import json
import numpy as np

from astropy import wcs
from astropy.io import fits

from sqlalchemy import ForeignKey, ForeignKeyConstraint, UniqueConstraint, Table
from sqlalchemy import Column, String, SmallInteger, Integer, Float, Date, DateTime, Enum, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_method

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

##############
# Base class #
##############


Base = declarative_base()

######################################################################
# Core database tables, track the data reduction through every step. #
######################################################################


class Workorder(Base):
    """ This class maps the 'workorders' table in the database.
    """

    __tablename__ = 'workorders'

    # Columns in the table.
    name = Column(String(100), primary_key=True)
    type = Column(String(6), nullable=False)
    date = Column(String(8), nullable=False)
    part = Column(SmallInteger)
    IHUID = Column(SmallInteger)
    FNUM = Column(Integer)
    completed = Column(Boolean, nullable=False, default=False)
    time_created = Column(DateTime, server_default=func.now())
    time_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def __init__(self, workorder: str) -> None:  # noqa
        """ Initialize the class for a specific workorder.

        Parameters
        ----------
        workorder : str
            The name of the workorder file.

        """

        workdict = utils.parse_workorder_filename(workorder)

        self.name = os.path.basename(workorder)
        self.type = workdict['type']
        self.date = workdict['date']
        self.part = workdict.get('part')
        self.IHUID = workdict.get('IHUID')
        self.FNUM = workdict.get('FNUM')

        return

    def __repr__(self) -> str:
        return f"<Workorder {self.name}>"


# Many-to-one association table between individual frames and image subtraction reference frames.
imgsub_association = Table("imgsub_association",
                           Base.metadata,
                           Column("IHUID", primary_key=True),
                           Column("FNUM", primary_key=True),
                           Column("refs_id", ForeignKey("imgsub_references.refs_id"), primary_key=True),
                           ForeignKeyConstraint(('IHUID', 'FNUM'),
                                                ('frames.IHUID', 'frames.FNUM')))


class Frame(Base):
    """ This class maps the 'frames' table in the database.
    """

    __tablename__ = 'frames'

    # Columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)
    frame_name = Column(String(100), unique=True, nullable=False)
    date_dir = Column(String(10), nullable=False)
    compression = Column(Enum('.gz', '.fz'))

    IMAGETYP = Column(String(20), nullable=False)
    OBJECT = Column(String(20), nullable=False)
    JD = Column(Float(53), nullable=False)
    datetime_obs = Column(DateTime, nullable=False)
    EXPTIME = Column(Float, nullable=False)
    CCDTEMP = Column(Float, nullable=False)

    NRA = Column(Float(53), nullable=False)
    NDEC = Column(Float(53), nullable=False)
    NHA = Column(Float(53), nullable=False)

    IHUVER = Column(Integer, nullable=False)
    CMID = Column(Integer, nullable=False)
    CMVER = Column(Integer, nullable=False)
    TELID = Column(Integer, nullable=False)
    TELVER = Column(Integer, nullable=False)
    FILID = Column(Integer, nullable=False)
    FILVER = Column(Integer, nullable=False)
    BIASVER = Column(Integer, nullable=False)
    DARKVER = Column(Integer, nullable=False)
    FLATVER = Column(Integer, nullable=False)
    TILTVER = Column(Integer, nullable=False)
    RDMODE = Column(Integer, nullable=False)
    AUTOGUID = Column(Boolean, nullable=False, default=False)
    MGENSTAT = Column(Float, nullable=False, default=0)

    # Relationships with other tables.
    # TODO set 'on delete' conditions?
    status = relationship("FrameStatus",
                          back_populates="frame",
                          uselist=False)
    quality = relationship('FrameQuality',
                           back_populates="frame",
                           uselist=False)
    calframe_quality = relationship('CalFrameQuality',
                                    back_populates="frame",
                                    uselist=False)
    subframe_quality = relationship('SubFrameQuality',
                                    back_populates="frame",
                                    uselist=False)
    psf = relationship('PSFStatistics',
                       back_populates="frame",
                       uselist=False)
    masterframes = relationship("MasterFrames",
                                back_populates="frame",
                                uselist=False)
    imgsubrefs = relationship("ImgSubReferences",
                              back_populates="imgsub_frames",
                              secondary=imgsub_association,
                              uselist=False)
    astrometry = relationship("Astrometry",
                              back_populates="frame",
                              uselist=False)
    aper_phot = relationship("AperPhotResult",
                             back_populates="frame",
                             uselist=False)
    tasks = relationship("TaskStatus",
                         back_populates="frame")
    queued_tasks = relationship("ProcessingQueue",
                                back_populates="frame",
                                cascade='all, delete-orphan')

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific frame.

        Parameters
        ----------
        filename : str
            The name of the frame.
        hdr : fits.Header or None
            The header of the frame.

        """

        frame_name, compression = utils.parse_filename(filename)
        date_dir = utils.find_path_component(filename, -3)

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.IHUID = hdr['IHUID']
        self.FNUM = hdr['FNUM']
        self.frame_name = frame_name
        self.compression = compression
        self.date_dir = date_dir

        # Basic properties of the frame.
        self.IMAGETYP = hdr['IMAGETYP']
        self.OBJECT = hdr['OBJECT']
        self.JD = hdr['JD']
        dtstr = ' '.join([hdr['DATE-OBS'], hdr['TIME-OBS']])
        self.datetime_obs = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S.%f")
        self.EXPTIME = hdr['EXPTIME']
        self.CCDTEMP = hdr['CCDTEMP']

        # Nominal pointing.
        self.NRA = hdr['NRA']
        self.NDEC = hdr['NDEC']
        self.NHA = hdr['NHA']

        # IDs and version numbers.
        self.IHUVER = hdr['IHUVER']
        self.CMID = hdr['CMID']
        self.CMVER = hdr['CMVER']
        self.TELID = hdr['TELID']
        self.TELVER = hdr['TELVER']
        self.FILID = hdr['FILID']
        self.FILVER = hdr['FILVER']
        self.BIASVER = hdr['BIASVER']
        self.DARKVER = hdr['DARKVER']
        self.FLATVER = hdr['FLATVER']
        self.TILTVER = hdr['TILTVER']
        self.RDMODE = hdr['RDMODE']
        self.AUTOGUID = hdr.get('AUTOGUID', False)
        self.MGENSTAT = hdr.get('MGENSTAT', 0)

        return

    @property
    def relpath(self) -> str:
        """ Return the relative path of the frame.
        """

        ihu_dir = f'ihu{self.IHUID:02d}'
        relpath = os.path.join(self.date_dir, ihu_dir, self.frame_name)

        if self.compression is not None:
            relpath = relpath + self.compression

        return relpath

    def abspath(self, root_dir: str) -> str:
        """ Return the absolute path of the frame.
        """

        return os.path.join(root_dir, self.relpath)

    @hybrid_method
    def distance(self, ra: float, dec: float) -> float:
        """ Compute the great circle distance between the frame center
            (NRA, NDEC) and a point (ra, dec).
        """

        ra0 = np.deg2rad(ra)
        dec0 = np.deg2rad(dec)
        ra1 = np.deg2rad(self.NRA*15)
        dec1 = np.deg2rad(self.NDEC)

        tmp = (np.sin(dec0) * np.sin(dec1) +
               np.cos(dec0) * np.cos(dec1) * np.cos(ra0 - ra1))

        return np.rad2deg(np.arccos(tmp))

    @distance.expression
    def distance(cls, ra, dec):  # noqa
        """ Database side computation of great circle distance.
        """

        ra0 = func.radians(ra)
        dec0 = func.radians(dec)
        ra1 = func.radians(cls.NRA*15)
        dec1 = func.radians(cls.NDEC)

        tmp = (func.sin(dec0) * func.sin(dec1) +
               func.cos(dec0) * func.cos(dec1) * func.cos(ra0 - ra1))

        return func.degrees(func.acos(tmp))

    def __repr__(self) -> str:
        return (f"<Frame"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM},"
                f" frame_name={self.frame_name},"
                f" date={self.datetime_obs},"
                f" imagetype={self.IMAGETYP}>")


class FrameStatus(Base):
    """ This class maps the 'frame_status' table in the database.
    """

    __tablename__ = 'frame_status'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Core columns.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)
    quality = Column(Integer, nullable=False, default=0)
    time_created = Column(DateTime, server_default=func.now())
    time_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Additional columns.
    task_type = Enum('success', 'failure', 'pending', 'N/A')
    task_calibrate = Column(task_type, nullable=False, default='N/A')
    task_astrometry = Column(task_type, nullable=False, default='N/A')
    task_aper_phot = Column(task_type, nullable=False, default='N/A')
    task_imgsub = Column(task_type, nullable=False, default='N/A')
    task_imgsub_phot = Column(task_type, nullable=False, default='N/A')

    saturation_masked = Column(Boolean, nullable=False, default=True)
    overscan_corrected = Column(Boolean, nullable=False, default=True)
    overscan_trimmed = Column(Boolean, nullable=False, default=True)
    bias_corrected = Column(Boolean, nullable=False, default=False)
    dark_corrected = Column(Boolean, nullable=False, default=False)
    flat_corrected = Column(Boolean, nullable=False, default=False)
    mask_added = Column(Boolean, nullable=False, default=False)

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="status")

    def __repr__(self) -> str:
        return (f"<FrameStatus"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM},"
                f" quality={self.quality}>")


class TaskStatus(Base):
    """ This class maps the 'task_status' table in the database.
    """

    __tablename__ = 'task_status'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    task_status = Enum('success', 'failure', 'pending', 'N/A')

    # Core columns.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)
    task = Column(String(100), primary_key=True)
    status = Column(task_status, nullable=False, default='N/A')
    time_created = Column(DateTime, server_default=func.now())
    time_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="tasks", uselist=False)

    def __repr__(self) -> str:
        return (f"<TaskStatus"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM},"
                f" task={self.task},"
                f" status={self.status}>")


class ProcessingQueue(Base):
    """ This class maps the 'processing_queue' table in the database.
    """

    __tablename__ = 'processing_queue'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)
    task = Column(String(100), primary_key=True)
    n_tries = Column(Integer, nullable=False, default=0)
    time_created = Column(DateTime, server_default=func.now())
    time_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="queued_tasks", uselist=False)

    def __repr__(self) -> str:
        return (f"<ProcessingQueue"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM},"
                f" task={self.task},"
                f" n_tries={self.n_tries}>")

#############################################################################
# Calibration tables, track the bias, dark, flat and masking of each frame. #
#############################################################################


class MasterFrames(Base):
    """ This class maps the 'master_frames' table in the database.
    """

    __tablename__ = 'master_frames'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)
    bias_id = Column(Integer, ForeignKey("masterbias.bias_id"), nullable=True)
    dark_id = Column(Integer, ForeignKey("masterdark.dark_id"), nullable=True)
    flat_id = Column(Integer, ForeignKey("masterflat.flat_id"), nullable=True)
    mask_id = Column(Integer, ForeignKey("mastermask.mask_id"), nullable=True)

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="masterframes")
    masterbias = relationship("MasterBias", back_populates="calibrated_frames")
    masterdark = relationship("MasterDark", back_populates="calibrated_frames")
    masterflat = relationship("MasterFlat", back_populates="calibrated_frames")
    mastermask = relationship("MasterMask", back_populates="calibrated_frames")

    def __repr__(self) -> str:
        return (f"<MasterFrames"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM},"
                f" bias_id={self.bias_id},"
                f" dark_id={self.dark_id},"
                f" flat_id={self.flat_id},"
                f" mask_id={self.mask_id}>")


# Many-to-many association table between individual bias frames and masterbiases.
bias_association = Table("bias_association",
                         Base.metadata,
                         Column("IHUID", primary_key=True),
                         Column("FNUM", primary_key=True),
                         Column("bias_id", ForeignKey("masterbias.bias_id"), primary_key=True),
                         ForeignKeyConstraint(('IHUID', 'FNUM'),
                                              ('frames.IHUID', 'frames.FNUM')))


class MasterBias(Base):
    """ This class maps the 'masterbias' table in the database.
    """

    __tablename__ = 'masterbias'

    # Core columns in the table.
    IHUID = Column(SmallInteger, nullable=False)
    bias_id = Column(Integer, primary_key=True)
    bias_name = Column(String(100), unique=True, nullable=False)
    year_dir = Column(String(4), nullable=False)
    compression = Column(Enum('.gz', '.fz'))
    quality = Column(Integer, nullable=False, default=0)
    DATEOBS = Column(Date, nullable=False)

    # Additional columns.
    CMID = Column(Integer, nullable=False)
    CMVER = Column(Integer, nullable=False)
    BIASVER = Column(Integer, nullable=False)
    RDMODE = Column(Integer, nullable=False)

    # Relationships with other tables.
    bias_frames = relationship("Frame", secondary=bias_association)
    calibrated_frames = relationship("MasterFrames", back_populates="masterbias")

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific masterbias.

        Parameters
        ----------
        filename : str
            The name of the masterbias.
        hdr : fits.Header or None
            The header of the masterbias.

        """

        bias_name, compression = utils.parse_filename(filename)
        year_dir = utils.find_path_component(filename, -3)

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Check that this is a masterbias.
        if hdr['IMAGETYP'] != 'masterbias':
            raise ValueError('Provided file is not a masterbias file.')

        # Set all fields.
        self.IHUID = hdr['IHUID']
        self.bias_name = bias_name
        self.compression = compression
        self.year_dir = year_dir
        self.DATEOBS = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d')

        self.CMID = hdr['CMID']
        self.CMVER = hdr['CMVER']
        self.BIASVER = hdr['BIASVER']
        self.RDMODE = hdr['RDMODE']

    @property
    def relpath(self) -> str:
        """ Return the relative path of the masterbias.
        """

        ihu_dir = f'ihu{self.IHUID:02d}'
        relpath = os.path.join(self.year_dir, ihu_dir, self.bias_name)

        if self.compression is not None:
            relpath = relpath + self.compression

        return relpath

    def abspath(self, root_dir: str) -> str:
        """ Return the absolute path of the masterbias.
        """

        return os.path.join(root_dir, self.relpath)

    def __repr__(self) -> str:
        return (f"<MasterBias"
                f" ihu={self.IHUID},"
                f" id={self.bias_id},"
                f" bias_name={self.bias_name},"
                f" date={self.DATEOBS}>")


# Many-to-many association table between individual dark frames and masterdarks.
dark_association = Table("dark_association",
                         Base.metadata,
                         Column("IHUID", primary_key=True),
                         Column("FNUM", primary_key=True),
                         Column("dark_id", ForeignKey("masterdark.dark_id"), primary_key=True),
                         ForeignKeyConstraint(('IHUID', 'FNUM'),
                                              ('frames.IHUID', 'frames.FNUM')))


class MasterDark(Base):
    """ This class maps the 'masterdark' table in the database.
    """

    __tablename__ = 'masterdark'

    # Core columns in the table.
    IHUID = Column(SmallInteger, nullable=False)
    dark_id = Column(Integer, primary_key=True)
    dark_name = Column(String(100), unique=True, nullable=False)
    year_dir = Column(String(4), nullable=False)
    compression = Column(Enum('.gz', '.fz'))
    quality = Column(Integer, nullable=False, default=0)
    DATEOBS = Column(Date, nullable=False)

    # Additional columns.
    CMID = Column(Integer, nullable=False)
    CMVER = Column(Integer, nullable=False)
    DARKVER = Column(Integer, nullable=False)
    RDMODE = Column(Integer, nullable=False)
    EXPTIME = Column(Float, nullable=True)
    MEDNTEMP = Column(Float, nullable=True)

    # Relationships with other tables.
    dark_frames = relationship("Frame", secondary=dark_association)
    calibrated_frames = relationship("MasterFrames", back_populates="masterdark")

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific masterdark.

        Parameters
        ----------
        filename : str
            The name of the masterdark.
        hdr : fits.Header or None
            The header of the masterdark.

        """

        dark_name, compression = utils.parse_filename(filename)
        year_dir = utils.find_path_component(filename, -3)

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Check that this is a masterdark.
        if hdr['IMAGETYP'] != 'masterdark':
            raise ValueError('Provided file is not a masterdark file.')

        # Set all fields.
        self.IHUID = hdr['IHUID']
        self.dark_name = dark_name
        self.compression = compression
        self.year_dir = year_dir
        self.DATEOBS = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d')

        self.CMID = hdr['CMID']
        self.CMVER = hdr['CMVER']
        self.DARKVER = hdr['DARKVER']
        self.RDMODE = hdr['RDMODE']
        self.EXPTIME = hdr['EXPTIME']
        self.MEDNTEMP = hdr['MEDNTEMP']

    @property
    def relpath(self) -> str:
        """ Return the relative path of the masterdark.
        """

        ihu_dir = f'ihu{self.IHUID:02d}'
        relpath = os.path.join(self.year_dir, ihu_dir, self.dark_name)

        if self.compression is not None:
            relpath = relpath + self.compression

        return relpath

    def abspath(self, root_dir: str) -> str:
        """ Return the absolute path of the masterdark.
        """

        return os.path.join(root_dir, self.relpath)

    def __repr__(self) -> str:
        return (f"<MasterDark"
                f" ihu={self.IHUID},"
                f" id={self.dark_id},"
                f" dark_name={self.dark_name},"
                f" date={self.DATEOBS}>")


# Many-to-many association table between individual flat frames and masterflats.
flat_association = Table("flat_association",
                         Base.metadata,
                         Column("IHUID", primary_key=True),
                         Column("FNUM", primary_key=True),
                         Column("flat_id", ForeignKey("masterflat.flat_id"), primary_key=True),
                         ForeignKeyConstraint(('IHUID', 'FNUM'),
                                              ('frames.IHUID', 'frames.FNUM')))


class MasterFlat(Base):
    """ This class maps the 'masterflat table in the database.
    """

    __tablename__ = 'masterflat'

    # Core columns in the table.
    IHUID = Column(SmallInteger, nullable=False)
    flat_id = Column(Integer, primary_key=True)
    flat_name = Column(String(100), unique=True, nullable=False)
    year_dir = Column(String(4), nullable=False)
    compression = Column(Enum('.gz', '.fz'))
    flat_type = Column(Enum('high', 'low'), nullable=False)
    quality = Column(Integer, nullable=False, default=0)
    DATEOBS = Column(Date, nullable=False)

    # Additional columns.
    CMID = Column(Integer, nullable=False)
    CMVER = Column(Integer, nullable=False)
    FLATVER = Column(Integer, nullable=False)
    RDMODE = Column(Integer, nullable=False)

    # Relationships with other tables.
    flat_frames = relationship("Frame", secondary=flat_association)
    calibrated_frames = relationship("MasterFrames", back_populates="masterflat")

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific masterflat.

        Parameters
        ----------
        filename : str
            The name of the masterflat.
        hdr : fits.Header or None
            The header of the masterflat.

        """

        flat_name, compression = utils.parse_filename(filename)
        year_dir = utils.find_path_component(filename, -3)

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Check that this is a masterflat.
        if hdr['IMAGETYP'] == 'masterflat':
            flat_type = 'high'
        elif hdr['IMAGETYP'] == 'lowmasterflat':
            flat_type = 'low'
        else:
            raise ValueError('Provided file is not a masterflat file.')

        # Set all fields.
        self.IHUID = hdr['IHUID']
        self.flat_name = flat_name
        self.compression = compression
        self.year_dir = year_dir
        self.flat_type = flat_type
        self.DATEOBS = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d')

        self.CMID = hdr['CMID']
        self.CMVER = hdr['CMVER']
        self.FLATVER = hdr['FLATVER']
        self.RDMODE = hdr['RDMODE']

    @property
    def relpath(self) -> str:
        """ Return the relative path of the masterflat.
        """

        ihu_dir = f'ihu{self.IHUID:02d}'
        relpath = os.path.join(self.year_dir, ihu_dir, self.flat_name)

        if self.compression is not None:
            relpath = relpath + self.compression

        return relpath

    def abspath(self, root_dir: str) -> str:
        """ Return the absolute path of the masterflat.
        """

        return os.path.join(root_dir, self.relpath)

    def __repr__(self) -> str:
        return (f"<MasterFlat"
                f" ihu={self.IHUID},"
                f" id={self.flat_id},"
                f" flat_name={self.flat_name},"
                f" date={self.DATEOBS}>")


class MasterMask(Base):
    """ This class maps the 'mastermask' table in the database.
    """

    __tablename__ = 'mastermask'

    # Core columns in the table.
    IHUID = Column(SmallInteger, nullable=False)
    mask_id = Column(Integer, primary_key=True)
    mask_name = Column(String(100), unique=True, nullable=False)
    year_dir = Column(String(4), nullable=False)
    compression = Column(Enum('.gz', '.fz'))
    quality = Column(Integer, nullable=False, default=0)
    DATEOBS = Column(Date, nullable=False)
    low_flat_id = Column(Integer, ForeignKey('masterflat.flat_id'), nullable=False)
    high_flat_id = Column(Integer, ForeignKey('masterflat.flat_id'), nullable=False)

    # Additinal columns.
    CMID = Column(Integer, nullable=False)
    CMVER = Column(Integer, nullable=False)
    FLATVER = Column(Integer, nullable=False)
    RDMODE = Column(Integer, nullable=False)

    # Relationships with other tables.
    low_masterflat = relationship("MasterFlat", foreign_keys=[low_flat_id])
    high_masterflat = relationship("MasterFlat", foreign_keys=[high_flat_id])
    calibrated_frames = relationship("MasterFrames", back_populates="mastermask")

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific mastermask.

        Parameters
        ----------
        filename : str
            The name of the mastermask.
        hdr : fits.Header or None
            The header of the mastermask.

        """

        mask_name, compression = utils.parse_filename(filename)
        year_dir = utils.find_path_component(filename, -3)

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Set all fields.
        self.IHUID = hdr['IHUID']
        self.mask_name = mask_name
        self.compression = compression
        self.year_dir = year_dir
        self.DATEOBS = datetime.strptime(hdr['DATE-OBS'], '%Y-%m-%d')

        self.CMID = hdr['CMID']
        self.CMVER = hdr['CMVER']
        self.FLATVER = hdr['FLATVER']
        self.RDMODE = hdr['RDMODE']

    @property
    def relpath(self) -> str:
        """ Return the relative path of the mastermask.
        """

        ihu_dir = f'ihu{self.IHUID:02d}'
        relpath = os.path.join(self.year_dir, ihu_dir, self.mask_name)

        if self.compression is not None:
            relpath = relpath + self.compression

        return relpath

    def abspath(self, root_dir: str) -> str:
        """ Return the absolute path of the mastermask.
        """

        return os.path.join(root_dir, self.relpath)

    def __repr__(self) -> str:
        return (f"<MasterMask"
                f" ihu={self.IHUID},"
                f" id={self.mask_id},"
                f" mask_name={self.mask_name},"
                f" date={self.DATEOBS}>")


#########################################################################
# Astrometry table, tracks the status of a frames astrometric solution. #
#########################################################################


class Astrometry(Base):
    """ This class maps the 'astrometry' table in the database.
    """

    __tablename__ = 'astrometry'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Core columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)

    # Quality parameters.
    program = Column(Enum('piastrom', 'hatpi_refine'), nullable=False)
    exit_code = Column(Integer, nullable=False)
    num_ids = Column(Integer, nullable=False)
    max_ids = Column(Integer, nullable=False)
    fit_err = Column(Float, nullable=False)
    pixel_scale = Column(Float, nullable=False)

    # WCS parameters.
    CRVAL1 = Column(Float(53))
    CRVAL2 = Column(Float(53))
    CRPIX1 = Column(Float(53))
    CRPIX2 = Column(Float(53))
    CD1_1 = Column(Float(53))
    CD1_2 = Column(Float(53))
    CD2_1 = Column(Float(53))
    CD2_2 = Column(Float(53))

    # SIP parameters.
    A_ORDER = Column(Integer)
    B_ORDER = Column(Integer)
    A = Column(String(1000))
    B = Column(String(1000))

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="astrometry")

    def __init__(self,  # noqa
                 filename: str,
                 program: str,
                 astromres: Any,  # TODO better type hint?
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific image frame.

        Parameters
        ----------
        filename : str
            The name of the image frame.
        program : str
            The name of the code used to solve the astrometry, should be either
            'piastrom' or 'hatpi_refine'.
        astromres : object
            The AstromRes instance returned by the program.
        hdr : fits.Header or None
            The header of the image frame.

        """

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.IHUID = hdr['IHUID']
        self.FNUM = hdr['FNUM']

        # Quality parameters.
        self.program = program
        self.exit_code = astromres.exit_code
        self.num_ids = astromres.num_ids
        self.max_ids = astromres.max_ids
        self.fit_err = astromres.err
        self.pixel_scale = astromres.pixel_scale

        # WCS parameters.
        self.CRVAL1 = hdr.get('CRVAL1')
        self.CRVAL2 = hdr.get('CRVAL2')
        self.CRPIX1 = hdr.get('CRPIX1')
        self.CRPIX2 = hdr.get('CRPIX2')
        self.CD1_1 = hdr.get('CD1_1')
        self.CD1_2 = hdr.get('CD1_2')
        self.CD2_1 = hdr.get('CD2_1')
        self.CD2_2 = hdr.get('CD2_2')

        # SIP parameters.
        w = wcs.WCS(hdr)
        if w.sip is not None:  # noqa: WCS does have sip attribute.
            self.A_ORDER = w.sip.a_order  # noqa
            self.B_ORDER = w.sip.b_order  # noqa
            self.A = json.dumps(w.sip.a.tolist())  # noqa
            self.B = json.dumps(w.sip.b.tolist())  # noqa

        return

    @property
    def wcs_a_array(self) -> Optional[np.ndarray]:
        """ Return the SIP A coefficients as an array.
        """

        if self.A is None:
            a_array = None
        else:
            a_array = np.array(json.loads(self.A))

        return a_array

    @property
    def wcs_b_array(self) -> Optional[np.ndarray]:
        """ Return the SIP B coefficients as an array.
        """

        if self.B is None:
            b_array = None
        else:
            b_array = np.array(json.loads(self.B))

        return b_array

    @hybrid_method
    def distance(self, ra: float, dec: float) -> float:
        """ Compute the great circle distance between the frame center
            (CRVAL1, CRVAL2) and a point (ra, dec).
        """

        ra0 = np.deg2rad(ra)
        dec0 = np.deg2rad(dec)
        ra1 = np.deg2rad(self.CRVAL1)
        dec1 = np.deg2rad(self.CRVAL2)

        tmp = (np.sin(dec0) * np.sin(dec1) +
               np.cos(dec0) * np.cos(dec1) * np.cos(ra0 - ra1))

        return np.rad2deg(np.arccos(tmp))

    @distance.expression
    def distance(cls, ra, dec):  # noqa
        """ Database side computation of great circle distance.
        """

        ra0 = func.radians(ra)
        dec0 = func.radians(dec)
        ra1 = func.radians(cls.CRVAL1)
        dec1 = func.radians(cls.CRVAL2)

        tmp = (func.sin(dec0) * func.sin(dec1) +
               func.cos(dec0) * func.cos(dec1) * func.cos(ra0 - ra1))

        return func.degrees(func.acos(tmp))

    def __repr__(self) -> str:
        return (f"<Astrometry"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM}, "
                f" program={self.program},"
                f" exit_code={self.exit_code}>")


class FrameQuality(Base):
    """ This class maps the 'frame_quality' table in the database.
    """

    __tablename__ = 'frame_quality'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Core columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)

    # Quality indicators from the headers.
    AIRMASS = Column(Float(53))
    SUNDIST = Column(Float(53))
    SUNELEV = Column(Float(53))
    MOONDIST = Column(Float(53))
    MOONELEV = Column(Float(53))
    MOONPH = Column(Float)
    WIND = Column(Float)
    AIRPRESS = Column(Float)
    HUMIDITY = Column(Float)
    SKYTDIFF = Column(Float)
    MNTTEMP1 = Column(Float)
    MNTTEMP2 = Column(Float)
    MNTTEMP3 = Column(Float)
    MNTTEMP4 = Column(Float)

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="quality")

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None,
                 ) -> None:
        """ Initialize the class for a specific frame.

        Parameters
        ----------
        filename : str
            The name of the frame.
        hdr : fits.Header or None
            The header of the frame.

        """

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.IHUID = hdr['IHUID']
        self.FNUM = hdr['FNUM']

        # Quality indicators from the headers.
        self.AIRMASS = hdr.get('X')  # Renamed AIRMASS for clarity.
        self.SUNDIST = hdr.get('SUNDIST')
        self.SUNELEV = hdr.get('SUNELEV')
        self.MOONDIST = hdr.get('MOONDIST')
        self.MOONELEV = hdr.get('MOONELEV')
        self.MOONPH = hdr.get('MOONPH')
        self.WIND = hdr.get('WIND')
        self.AIRPRESS = hdr.get('AIRPRESS')
        self.HUMIDITY = hdr.get('HUMIDITY')
        self.SKYTDIFF = hdr.get('SKYTDIFF')
        self.MNTTEMP1 = hdr.get('MNTTEMP1')
        self.MNTTEMP2 = hdr.get('MNTTEMP2')
        self.MNTTEMP3 = hdr.get('MNTTEMP3')
        self.MNTTEMP4 = hdr.get('MNTTEMP4')

        return

    def __repr__(self) -> str:
        return (f"<FrameQuality"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM}>")


class CalFrameQuality(Base):
    """ This class maps the 'calframe_quality' table in the database.
    """

    __tablename__ = 'calframe_quality'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Core columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)

    # Quality indicators computed from the calibrated image.
    calframe_mean = Column(Float)
    calframe_stdev = Column(Float)
    calframe_median = Column(Float)
    calframe_mad = Column(Float)
    calframe_p05 = Column(Float)
    calframe_p95 = Column(Float)

    calframe_tile_medians_mean = Column(Float)
    calframe_tile_medians_stdev = Column(Float)
    calframe_tile_medians_median = Column(Float)
    calframe_tile_medians_mad = Column(Float)

    # Other quality indcators.
    calframe_masked = Column(Integer)
    calframe_fault = Column(Integer)
    calframe_hot = Column(Integer)
    calframe_cosmic = Column(Integer)
    calframe_outer = Column(Integer)
    calframe_oversaturated = Column(Integer)
    calframe_leaked = Column(Integer)
    calframe_saturated = Column(Integer)
    calframe_interpolated = Column(Integer)

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="calframe_quality")

    def __init__(self,  # noqa
                 filename: str,
                 fits_info: dict[str, Any],
                 mask_counts: dict[str, Any],
                 hdr: Optional[fits.Header] = None,
                 ) -> None:
        """ Initialize the class for a specific frame.

        Parameters
        ----------
        filename : str
            The name of the frame.
        fits_info : dict
            A dictionary containing image statistics.
        mask_counts : dict
            A dictionary containing bad pixel statistics.
        hdr : fits.Header or None
            The header of the frame.

        """

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.IHUID = hdr['IHUID']
        self.FNUM = hdr['FNUM']

        # Quality indicators computed from the calibrated image.
        self.calframe_mean = fits_info['overall_mean']
        self.calframe_stdev = fits_info['overall_stdev']
        self.calframe_median = fits_info['overall_median']
        self.calframe_mad = fits_info['overall_mad']
        self.calframe_p05 = fits_info['overall_percentiles'][0]
        self.calframe_p95 = fits_info['overall_percentiles'][1]

        self.calframe_tile_medians_mean = fits_info['tile_medians_mean']
        self.calframe_tile_medians_stdev = fits_info['tile_medians_stdev']
        self.calframe_tile_medians_median = fits_info['tile_medians_median']
        self.calframe_tile_medians_mad = fits_info['tile_medians_mad']

        self.calframe_masked = hdr['NAXIS1']*hdr['NAXIS2'] - mask_counts['good']
        self.calframe_fault = mask_counts['fault']
        self.calframe_hot = mask_counts['hot']
        self.calframe_cosmic = mask_counts['cosmic']
        self.calframe_outer = mask_counts['outer']
        self.calframe_oversaturated = mask_counts['oversaturated']
        self.calframe_leaked = mask_counts['leaked']
        self.calframe_saturated = mask_counts['saturated']
        self.calframe_interpolated = mask_counts['interpolated']

        return

    def __repr__(self) -> str:
        return (f"<CalFrameQuality"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM}>")


class SubFrameQuality(Base):
    """ This class maps the 'subframe_quality' table in the database.
    """

    __tablename__ = 'subframe_quality'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Core columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)

    # Quality indicators computed from the subtracted image.
    subframe_mean = Column(Float)
    subframe_stdev = Column(Float)
    subframe_median = Column(Float)
    subframe_mad = Column(Float)
    subframe_p05 = Column(Float)
    subframe_p95 = Column(Float)

    subframe_tile_medians_mean = Column(Float)
    subframe_tile_medians_stdev = Column(Float)
    subframe_tile_medians_median = Column(Float)
    subframe_tile_medians_mad = Column(Float)

    # Other quality indcators.
    subframe_masked = Column(Integer)
    subframe_fault = Column(Integer)
    subframe_hot = Column(Integer)
    subframe_cosmic = Column(Integer)
    subframe_outer = Column(Integer)
    subframe_oversaturated = Column(Integer)
    subframe_leaked = Column(Integer)
    subframe_saturated = Column(Integer)
    subframe_interpolated = Column(Integer)

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="subframe_quality")

    def __init__(self,  # noqa
                 filename: str,
                 fits_info: dict[str, Any],
                 mask_counts: dict[str, Any],
                 hdr: Optional[fits.Header] = None,
                 ) -> None:
        """ Initialize the class for a specific frame.

        Parameters
        ----------
        filename : str
            The name of the frame.
        fits_info : dict
            A dictionary containing image statistics.
        mask_counts : dict
            A dictionary containing bad pixel statistics.
        hdr : fits.Header or None
            The header of the frame.

        """

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.IHUID = hdr['IHUID']
        self.FNUM = hdr['FNUM']

        # Quality indicators computed from the subtracted image.
        self.subframe_mean = fits_info['overall_mean']
        self.subframe_stdev = fits_info['overall_stdev']
        self.subframe_median = fits_info['overall_median']
        self.subframe_mad = fits_info['overall_mad']
        self.subframe_p05 = fits_info['overall_percentiles'][0]
        self.subframe_p95 = fits_info['overall_percentiles'][1]

        self.subframe_tile_medians_mean = fits_info['tile_medians_mean']
        self.subframe_tile_medians_stdev = fits_info['tile_medians_stdev']
        self.subframe_tile_medians_median = fits_info['tile_medians_median']
        self.subframe_tile_medians_mad = fits_info['tile_medians_mad']

        self.subframe_masked = hdr['NAXIS1']*hdr['NAXIS2'] - mask_counts['good']
        self.subframe_fault = mask_counts['fault']
        self.subframe_hot = mask_counts['hot']
        self.subframe_cosmic = mask_counts['cosmic']
        self.subframe_outer = mask_counts['outer']
        self.subframe_oversaturated = mask_counts['oversaturated']
        self.subframe_leaked = mask_counts['leaked']
        self.subframe_saturated = mask_counts['saturated']
        self.subframe_interpolated = mask_counts['interpolated']

        return

    def __repr__(self) -> str:
        return (f"<SubFrameQuality"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM}>")


class PSFStatistics(Base):
    """ This class maps the 'psf_statistics' table in the database.
    """

    __tablename__ = 'psf_statistics'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Core columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)

    # S, D, K and other fistar parameters.
    S_median = Column(Float)
    D_median = Column(Float)
    K_median = Column(Float)
    S_grid_median = Column(String(1000))
    D_grid_median = Column(String(1000))
    K_grid_median = Column(String(1000))
    FWHM_median = Column(Float)
    Ellip_median = Column(Float)
    PA_median = Column(Float)

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="psf")

    def __init__(self,  # noqa
                 filename: str,
                 psfstats: dict[str, Any],
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific image frame.

        Parameters
        ----------
        filename: str
            The name of the image frame.
        psfstats: dict
            A dictionary containing PSF statistics.
        hdr: fits.Header or None
            The header of the image frame.

        """

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.IHUID = hdr['IHUID']
        self.FNUM = hdr['FNUM']

        # S, D, K and other fistar parameters.
        self.S_median = psfstats['S_median']
        self.D_median = psfstats['D_median']
        self.K_median = psfstats['K_median']
        self.S_grid_median = json.dumps(psfstats['S_grid_median'].tolist())
        self.D_grid_median = json.dumps(psfstats['D_grid_median'].tolist())
        self.K_grid_median = json.dumps(psfstats['K_grid_median'].tolist())
        self.FWHM_median = psfstats['FWHM_median']
        self.Ellip_median = psfstats['Ellip_median']
        self.PA_median = psfstats['PA_median']

    @property
    def s_grid_array(self) -> Optional[np.ndarray]:
        """ Return the grid of S values as an array.
        """

        if self.S_grid_median is None:
            array = None
        else:
            array = np.array(json.loads(self.S_grid_median))

        return array

    @property
    def d_grid_array(self) -> Optional[np.ndarray]:
        """ Return the grid of D values as an array.
        """

        if self.S_grid_median is None:
            array = None
        else:
            array = np.array(json.loads(self.D_grid_median))

        return array

    @property
    def k_grid_array(self) -> Optional[np.ndarray]:
        """ Return the grid of K values as an array.
        """

        if self.S_grid_median is None:
            array = None
        else:
            array = np.array(json.loads(self.K_grid_median))

        return array

    def __repr__(self) -> str:
        return (f"<PSFStatistics"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM},"
                f" S={self.S_median:.2f},"
                f" D={self.D_median:.2f},"
                f" K={self.K_median:.2f}>")


##########################################################################
#  Photometry tables, track the catalogs and extracted photometry files  #
##########################################################################


class StarCatalog(Base):
    """ This class maps the 'star_catalogs' table in the database.
    """

    __tablename__ = 'star_catalogs'
    __table_args__ = (UniqueConstraint('catalog_dir', 'catalog_name', name='catalog_path'),
                      )

    # Core columns in the table.
    catalog_id = Column(Integer, primary_key=True)
    catalog_name = Column(String(100), nullable=False)
    catalog_dir = Column(String(100), nullable=False)
    compression = Column(Enum('.gz', '.fz'))
    time_created = Column(DateTime, server_default=func.now())
    time_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Additional columns.
    RA = Column(Float(53), nullable=False)
    DEC = Column(Float(53), nullable=False)
    SIZE = Column(Float, nullable=False)
    TYPE = Column(String(4), nullable=False)
    MAGBRIGH = Column(Float, nullable=False)
    MAGFAINT = Column(Float, nullable=False)
    EPOCH = Column(Float, nullable=False)
    SOURCE = Column(String(20), nullable=False)
    CONTRAD = Column(Float, nullable=False)
    OBJECT = Column(String(20))

    # Relationships with other tables.
    aper_phot_files = relationship('AperPhotResult', back_populates='catalog')

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific star catalog.

        Parameters
        ----------
        filename : str
            The name of the star catalog file.
        hdr: fits.Header or None
            The header of the star catalog file.

        """

        catalog_name, compression = utils.parse_filename(filename)
        catalog_dir = utils.find_path_component(filename, -2)

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.catalog_name = catalog_name
        self.compression = compression
        self.catalog_dir = catalog_dir

        # Additional columns.
        self.RA = hdr['RA']
        self.DEC = hdr['DEC']
        self.SIZE = hdr['SIZE']
        self.TYPE = hdr['TYPE']
        self.MAGBRIGH = hdr['MAGBRIGH']
        self.MAGFAINT = hdr['MAGFAINT']
        self.EPOCH = hdr['EPOCH']
        self.SOURCE = hdr['SOURCE']
        self.CONTRAD = hdr['CONTRAD']
        self.OBJECT = hdr.get('OBJECT')

        return

    @property
    def relpath(self) -> str:
        """ Return the relative path of the star catalog.
        """

        relpath = os.path.join(self.catalog_dir, self.catalog_name)

        if self.compression is not None:
            relpath = relpath + self.compression

        return relpath

    def abspath(self, root_dir: str) -> str:
        """ Return the absolute path of the star catalog.
        """

        return os.path.join(root_dir, self.relpath)

    @hybrid_method
    def distance(self, ra: float, dec: float) -> float:
        """ Compute the great circle distance between the catalog center
            (RA, DEC) and a point (ra, dec).
        """

        ra0 = np.deg2rad(ra)
        dec0 = np.deg2rad(dec)
        ra1 = np.deg2rad(self.RA)
        dec1 = np.deg2rad(self.DEC)

        tmp = (np.sin(dec0) * np.sin(dec1) +
               np.cos(dec0) * np.cos(dec1) * np.cos(ra0 - ra1))

        return np.rad2deg(np.arccos(tmp))

    @distance.expression
    def distance(cls, ra, dec):  # noqa
        """ Database side computation of great circle distance.
        """

        ra0 = func.radians(ra)
        dec0 = func.radians(dec)
        ra1 = func.radians(cls.RA)
        dec1 = func.radians(cls.DEC)

        tmp = (func.sin(dec0) * func.sin(dec1) +
               func.cos(dec0) * func.cos(dec1) * func.cos(ra0 - ra1))

        return func.degrees(func.acos(tmp))

    def __repr__(self) -> str:
        return (f"<StarCatalog"
                f" relpath={self.relpath}>")


class AperPhotResult(Base):
    """ This class maps the 'aper_phot_results' table in the database.
    """

    __tablename__ = 'aper_phot_results'
    __table_args__ = (ForeignKeyConstraint(("IHUID", "FNUM"),
                                           ("frames.IHUID", "frames.FNUM")),
                      )

    # Core columns in the table.
    IHUID = Column(SmallInteger, primary_key=True)
    FNUM = Column(Integer, primary_key=True)
    OBJECT = Column(String(20), nullable=False)
    file_name = Column(String(100), unique=True, nullable=False)
    date_dir = Column(String(10), nullable=False)
    compression = Column(Enum('.gz', '.fz'))
    catalog_id = Column(Integer, ForeignKey("star_catalogs.catalog_id"), nullable=False)

    # Additional columns.
    NSTARS = Column(Integer, nullable=False)
    NAPER = Column(Integer, nullable=False)
    APRAD = Column(String(200), nullable=False)
    ANRAD = Column(String(200), nullable=False)
    ANWID = Column(String(200), nullable=False)
    MAGDIFF = Column(String(200), nullable=True)

    # Relationships with other tables.
    frame = relationship("Frame", back_populates="aper_phot")
    catalog = relationship("StarCatalog", back_populates="aper_phot_files")

    def __init__(self,  # noqa
                 filename: str,
                 hdr: Optional[fits.Header] = None
                 ) -> None:
        """ Initialize the class for a specific aperture phtometry file.

        Parameters
        ----------
        filename : str
            The name of the aperture photometry file.
        hdr : fits.Header or None
            The header of the aperture photometry file.

        """

        file_name, compression = utils.parse_filename(filename)
        date_dir = utils.find_path_component(filename, -3)

        # Read the header information.
        if hdr is None:
            hdr = fits.getheader(filename)

        # Core columns.
        self.IHUID = hdr['IHUID']
        self.FNUM = hdr['FNUM']
        self.OBJECT = hdr['OBJECT']
        self.file_name = file_name
        self.compression = compression
        self.date_dir = date_dir

        # Additional columns.
        self.NSTARS = hdr['NSTARS']
        self.NAPER = hdr['NAPER']
        self.APRAD = json.dumps([hdr[f'APRAD{i}'] for i in range(hdr['NAPER'])])
        self.ANRAD = json.dumps([hdr[f'ANRAD{i}'] for i in range(hdr['NAPER'])])
        self.ANWID = json.dumps([hdr[f'ANWID{i}'] for i in range(hdr['NAPER'])])
        self.MAGDIFF = json.dumps([hdr[f'MAGDIFF{i}'] for i in range(hdr['NAPER'])])

        return

    @property
    def aperture_radii(self) -> np.ndarray:
        """ Return the aperture radii as an array.
        """

        return np.array(json.loads(self.APRAD))

    @property
    def annulus_radii(self) -> np.ndarray:
        """ Return the annulus radii as an array.
        """

        return np.array(json.loads(self.ANRAD))

    @property
    def annulus_widths(self) -> np.ndarray:
        """ Return the annlus widths as an array.
        """

        return np.array(json.loads(self.ANWID))

    @property
    def magnitude_offsets(self) -> np.ndarray:
        """ Return the magnitude offsets as an array.
        """

        return np.array(json.loads(self.MAGDIFF))

    @property
    def relpath(self):
        """ Return the relative path of the aperture photometry file.
        """

        ihu_dir = f'ihu{self.IHUID:02d}'
        relpath = os.path.join(self.date_dir, ihu_dir, self.file_name)

        if self.compression is not None:
            relpath = relpath + self.compression

        return relpath

    def abspath(self, root_dir: str) -> str:
        """ Return the absolute path of the aperture photometry file.
        """

        return os.path.join(root_dir, self.relpath)

    def __repr__(self) -> str:
        return (f"<AperPhotResult"
                f" ihu={self.IHUID},"
                f" frame={self.FNUM},"
                f" relpath={self.relpath}>")

###############################
#  Image subtraction tables.  #
###############################


# One-to-many association table between individual frames and astrometric reference frames.
astroref_association = Table("astroref_association",
                             Base.metadata,
                             Column("IHUID", primary_key=True),
                             Column("FNUM", primary_key=True),
                             Column("refs_id", ForeignKey("imgsub_references.refs_id"), primary_key=True),
                             ForeignKeyConstraint(('IHUID', 'FNUM'),
                                                  ('frames.IHUID', 'frames.FNUM')))


# Many-to-many association table between individual frames and photometric reference frames.
photoref_association = Table("photoref_association",
                             Base.metadata,
                             Column("IHUID", primary_key=True),
                             Column("FNUM", primary_key=True),
                             Column("refs_id", ForeignKey("imgsub_references.refs_id"), primary_key=True),
                             ForeignKeyConstraint(('IHUID', 'FNUM'),
                                                  ('frames.IHUID', 'frames.FNUM')))


class ImgSubReferences(Base):
    """ This class maps the 'imgsub_references' table in the database.
    """

    __tablename__ = 'imgsub_references'

    # Core columns in the table.
    refs_id = Column(Integer, primary_key=True)
    IHUID = Column(SmallInteger)
    OBJECT = Column(String(20), nullable=False)
    refs_dir = Column(String(100), nullable=False, unique=True)
    refs_type = Column(Enum('OBJECT_IHUID', 'OBJECT_ONLY'), nullable=False)
    refs_version = Column(SmallInteger, nullable=False)
    astroref_name = Column(String(100), nullable=False)
    photoref_name = Column(String(100), nullable=False)
    quality = Column(Integer, nullable=False, default=0)
    time_created = Column(DateTime, server_default=func.now())
    time_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships with other tables.
    imgsub_frames = relationship("Frame", back_populates='imgsubrefs', secondary=imgsub_association)
    astroref_frame = relationship("Frame", secondary=astroref_association, uselist=False)
    photoref_frames = relationship("Frame", secondary=photoref_association)

    def __init__(self,  # noqa
                 refs_path: str
                 ) -> None:
        """ Initialize the class for a specific aperture phtometry file.

        Parameters
        ----------
        refs_path: str
            Full path to the set of reference files.

        """

        # Parse the file path to extract information.
        refsdict, refs_dir = utils.parse_imgsub_references_path(refs_path)

        # Find the astrometric reference files.
        astroref_glob = glob.glob(os.path.join(refs_path, 'astroref_*_??.fits'))
        if len(astroref_glob) == 1:
            astroref_name = astroref_glob[0]
        else:
            raise ValueError('Too many files found matching astroref_*.fits, expected one.')

        # Check that the fistar and regions file also exist.
        fistar_file = astroref_name.replace('.fits', '.fistar.fits')
        if not os.path.exists(fistar_file):
            raise ValueError('Could not find astroref_*.fistar.fits file.')

        regions_file = astroref_name.replace('.fits', '.reg')
        if not os.path.exists(regions_file):
            raise ValueError('Could not find astroref_*.reg file.')

        # Find the photometric reference file.
        photoref_glob = glob.glob(os.path.join(refs_path, 'photoref_*.fits'))
        if len(photoref_glob) == 1:
            photoref_name = photoref_glob[0]
        else:
            raise ValueError('Too many files found matching photoref_*.fits, expected one.')

        # Parse the astrometric and photometric file names.
        astroref_name, compression = utils.parse_filename(astroref_name)
        photoref_name, compression = utils.parse_filename(photoref_name)

        # Core columns.
        self.IHUID = refsdict['ihuid']
        self.OBJECT = refsdict['object']
        self.refs_dir = refs_dir
        self.refs_type = refsdict['type']
        self.refs_version = refsdict['version']
        self.astroref_name = astroref_name
        self.photoref_name = photoref_name

        return

    def relpath(self, filetype: str) -> str:
        """ Return the relative path of the specified reference file.

        Parameters
        ----------
        filetype : str
            Which reference file to produce the path for, can be astroref,
            photoref, astroref_fistar or astroref_regions.

        Returns
        -------
        relpath : str
            The relative path to the specified reference file.

        """

        if filetype == 'photoref':
            filename = self.photoref_name
        elif filetype == 'astroref':
            filename = self.astroref_name
        elif filetype == 'astroref_fistar':
            filename = self.astroref_name.replace('.fits', '.fistar.fits')
        elif filetype == 'astroref_regions':
            filename = self.astroref_name.replace('.fits', '.reg')
        else:
            raise ValueError(f"Unknown value '{filetype}', for parameter filetype.")

        return os.path.join(self.refs_dir, filename)

    def abspath(self, filetype: str, root_dir: str) -> str:
        """ Return the absolute path for the specified reference file.
        """

        return os.path.join(root_dir, self.relpath(filetype))

    def __repr__(self) -> str:
        return (f"<ImgSubReferences"
                f" ihu={self.IHUID},"
                f" object={self.OBJECT},"
                f" refs_type={self.refs_type},"
                f" refs_version={self.refs_version},"
                f" astroref_name={self.astroref_name},"
                f" photoref_name={self.photoref_name}>")


def main():

    return


if __name__ == '__main__':
    main()
