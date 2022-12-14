import logging
from typing import Optional, Union
from datetime import datetime

from sqlalchemy import select

from astropy.table import Table

from . import utils
from .models import Frame, FrameQuality, CalFrameQuality
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

###############################
#  REFERENCE FRAME SELECTION  #
###############################


def select_reference_frames(field_name: str,
                            ihu_id: Optional[int] = None,
                            daterange: Optional[Union[tuple[datetime, datetime], list[datetime]]] = None,
                            maxsunelev: Optional[float] = -18,
                            maxmoonelev: Optional[float] = 0,
                            minframes: int = 100
                            ) -> Optional[Table]:
    """ Selects a list of potential photometric reference frames that can be
        used to generate a master photometric reference frame.

    Parameters
    -----------
    field_name : str
        value for Frame.OBJECT
    ihu_id : int or None
        value of Frame.IHUID
    daterange : list or tuple or None
        Force Frame.datetime_obs to the range [datetime_min, datetime_max]
    maxsunelev : float or None
        Maximum sun elevation in degrees.
    maxmoonelev : float or None
        Maximum moon elevation in degrees.
    minframes : int
        The minimumn number of frames returned by the query.

    Returns
    --------
     tab : astropy.table.Table
        Table containing database information on the frames that might be used
        to create a master photometric reference, with the best frames having
        the smallest value for the 'photo_dist' column.

    """

    with Session() as session:

        # Perform the selection
        statement = select([*Frame.__table__.columns,
                            *Astrometry.__table__.columns,
                            *PSFStatistics.__table__.columns,
                            *FrameQuality.__table__.columns,
                            *CalFrameQuality.__table__.columns,
                            *AperPhotResult.__table__.columns])

        statement = statement.join(Astrometry)
        statement = statement.join(PSFStatistics)
        statement = statement.join(FrameQuality)
        statement = statement.join(CalFrameQuality)
        statement = statement.join(AperPhotResult)

        statement = statement.where(Frame.OBJECT == field_name)
        statement = statement.where(Astrometry.exit_code == 0)
        statement = statement.where(AperPhotResult.MAGDIFF != None)  # noqa

        if ihu_id is not None:
            statement = statement.where(Frame.IHUID == ihu_id)
        if daterange is not None:
            statement = statement.filter(Frame.datetime_obs >= daterange[0],
                                         Frame.datetime_obs <= daterange[1])
        if maxsunelev is not None:
            statement = statement.where(FrameQuality.SUNELEV < maxsunelev)
        if maxmoonelev is not None:
            statement = statement.where(FrameQuality.MOONELEV < maxmoonelev)

        # Execute the query.
        result = session.execute(statement)

        rows = result.all()
        names = result.keys()

    # Check that a reasonable number of frames were found.
    nframes = len(rows)
    if nframes < minframes:
        LOGWARNING("Too few frames found for selecting photometric references.")
        return None

    # Put the results in a table.
    tab = Table(rows=rows, names=names)

    # Convert the MAGDIFF column to an array.
    tab['MAGDIFF'] = utils.column_json_to_array(tab['MAGDIFF'])

    # Sort the table.
    tab.sort(['IHUID', 'FNUM'])

    return tab


def main():

    return


if __name__ == '__main__':
    main()
