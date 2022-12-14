import logging

from sqlalchemy import select, func
from sqlalchemy.sql.expression import text

from astropy.table import Table

from . import utils
from .models import Frame, TaskStatus, FrameQuality, CalFrameQuality
from .models import Astrometry, PSFStatistics, AperPhotResult
from .methods import Session

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
#  Graphics  #
##############

def create_graphics_for(num_days=1):

    ref_time = func.date_sub(func.now(), text(f'interval {num_days} day'))

    with Session() as session:

        statement = select(Frame.date_dir, Frame.IHUID)
        statement = statement.join(Frame.tasks)
        statement = statement.where(TaskStatus.time_updated > ref_time)
        statement = statement.group_by(Frame.date_dir, Frame.IHUID)

        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    if len(rows) == 0:
        LOGWARNING("Query returned no results.")
        return [], []

    tab = Table(rows=rows, names=names)

    return tab['date_dir'].value, tab['IHUID'].value


def get_night_metrics(date_dir):

    with Session() as session:

        statement = select(Frame.JD,
                           Frame.FNUM,
                           Frame.OBJECT,
                           Frame.IMAGETYP,
                           Frame.MGENSTAT,
                           FrameQuality.WIND,
                           FrameQuality.AIRPRESS,
                           FrameQuality.HUMIDITY,
                           FrameQuality.SKYTDIFF,
                           FrameQuality.SUNELEV,
                           FrameQuality.MOONELEV,
                           FrameQuality.MOONPH,
                           func.ifnull(FrameQuality.MNTTEMP1, 0).label('MNTTEMP1'),
                           func.ifnull(FrameQuality.MNTTEMP2, 0).label('MNTTEMP2'),
                           func.ifnull(FrameQuality.MNTTEMP3, 0).label('MNTTEMP3'),
                           func.ifnull(FrameQuality.MNTTEMP4, 0).label('MNTTEMP4'))
        statement = statement.join(Frame.quality)
        statement = statement.where(Frame.date_dir == date_dir)
        statement = statement.order_by(Frame.IHUID, Frame.FNUM)

        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    if len(rows) == 0:
        LOGWARNING("Query returned 0 frames.")
        return None

    tab = Table(rows=rows, names=names)

    return tab


def get_ihu_metrics(date_dir, ihu_id=None):

    with Session() as session:

        statement = select(Frame.JD,
                           Frame.IHUID,
                           Frame.FNUM,
                           Frame.OBJECT,
                           Frame.EXPTIME,
                           Frame.CCDTEMP,
                           Frame.IMAGETYP,
                           FrameQuality.SUNDIST,
                           FrameQuality.MOONDIST,
                           FrameQuality.AIRMASS)
        statement = statement.join(Frame.quality)
        statement = statement.where(Frame.date_dir == date_dir)
        if ihu_id is not None:
            statement = statement.where(Frame.IHUID == ihu_id)
        statement = statement.order_by(Frame.IHUID, Frame.FNUM)

        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    if len(rows) == 0:
        LOGWARNING("Query returned 0 frames.")
        return None

    tab = Table(rows=rows, names=names)

    return tab


def get_image_statistics(date_dir, ihu_id=None):

    with Session() as session:

        statement = select(Frame.JD,
                           Frame.IHUID,
                           Frame.FNUM,
                           Frame.OBJECT,
                           Frame.IMAGETYP,
                           CalFrameQuality.calframe_masked,
                           CalFrameQuality.calframe_saturated,
                           CalFrameQuality.calframe_oversaturated,
                           CalFrameQuality.calframe_leaked,
                           CalFrameQuality.calframe_mean,
                           CalFrameQuality.calframe_median,
                           CalFrameQuality.calframe_stdev,
                           CalFrameQuality.calframe_mad,
                           CalFrameQuality.calframe_p05,
                           CalFrameQuality.calframe_p95,
                           CalFrameQuality.calframe_tile_medians_mean,
                           CalFrameQuality.calframe_tile_medians_median,
                           CalFrameQuality.calframe_tile_medians_stdev,
                           CalFrameQuality.calframe_tile_medians_mad)
        statement = statement.join(Frame.calframe_quality)
        statement = statement.where(Frame.date_dir == date_dir)
        if ihu_id is not None:
            statement = statement.where(Frame.IHUID == ihu_id)
        statement = statement.order_by(Frame.IHUID, Frame.FNUM)

        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    if len(rows) == 0:
        LOGWARNING("Query returned 0 frames.")
        return None

    tab = Table(rows=rows, names=names)

    return tab


def get_astrometry(date_dir, ihu_id=None):

    with Session() as session:

        statement = select(Frame.JD,
                           Frame.IHUID,
                           Frame.FNUM,
                           Frame.OBJECT,
                           Frame.IMAGETYP,
                           Frame.NRA,
                           Frame.NDEC,
                           Astrometry.num_ids,
                           Astrometry.max_ids,
                           Astrometry.fit_err,
                           Astrometry.pixel_scale,
                           Astrometry.CRVAL1,
                           Astrometry.CRVAL2,
                           Astrometry.CD1_1,
                           Astrometry.CD1_2,
                           Astrometry.CD2_1,
                           Astrometry.CD2_2,
                           Astrometry.A,
                           Astrometry.B)
        statement = statement.join(Frame.astrometry)
        statement = statement.where(Frame.date_dir == date_dir,
                                    Astrometry.exit_code == 0)
        if ihu_id is not None:
            statement = statement.where(Frame.IHUID == ihu_id)
        statement = statement.order_by(Frame.IHUID, Frame.FNUM)

        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    if len(rows) == 0:
        LOGWARNING("Query returned 0 frames.")
        return None

    tab = Table(rows=rows, names=names)

    tab['A'] = utils.column_json_to_array(tab['A'])
    tab['B'] = utils.column_json_to_array(tab['B'])

    return tab


def get_psf_statistics(date_dir, ihu_id=None):

    with Session() as session:

        statement = select(Frame.JD,
                           Frame.IHUID,
                           Frame.FNUM,
                           Frame.OBJECT,
                           Frame.IMAGETYP,
                           PSFStatistics.S_median,
                           PSFStatistics.D_median,
                           PSFStatistics.K_median,
                           PSFStatistics.S_grid_median,
                           PSFStatistics.D_grid_median,
                           PSFStatistics.K_grid_median)
        statement = statement.join(Frame.psf)
        statement = statement.where(Frame.date_dir == date_dir)
        if ihu_id is not None:
            statement = statement.where(Frame.IHUID == ihu_id)
        statement = statement.order_by(Frame.IHUID, Frame.FNUM)

        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    if len(rows) == 0:
        LOGWARNING("Query returned 0 frames.")
        return None

    tab = Table(rows=rows, names=names)

    tab['S_grid_median'] = utils.column_json_to_array(tab['S_grid_median'])
    tab['D_grid_median'] = utils.column_json_to_array(tab['D_grid_median'])
    tab['K_grid_median'] = utils.column_json_to_array(tab['K_grid_median'])

    return tab


def get_aperphot_metrics(date_dir, ihu_id=None):

    with Session() as session:

        statement = select(Frame.JD,
                           Frame.IHUID,
                           Frame.FNUM,
                           Frame.OBJECT,
                           Frame.IMAGETYP,
                           AperPhotResult.MAGDIFF)
        statement = statement.join(Frame.aper_phot)
        statement = statement.where(Frame.date_dir == date_dir)
        statement = statement.where(AperPhotResult.MAGDIFF != None)  # noqa
        if ihu_id is not None:
            statement = statement.where(Frame.IHUID == ihu_id)
        statement = statement.order_by(Frame.IHUID, Frame.FNUM)

        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    if len(rows) == 0:
        LOGWARNING("Query returned 0 frames.")
        return None

    tab = Table(rows=rows, names=names)

    tab['MAGDIFF'] = utils.column_json_to_array(tab['MAGDIFF'])

    return tab


def main():

    return


if __name__ == '__main__':
    main()
