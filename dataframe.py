"""
DataFrame
---------
An efficient 2D container for potentially mixed-type time series or other
labeled data series.

Similar to its R counterpart, data.frame, except providing automatic data
alignment and a host of useful data manipulation methods having to do with the
labeling information
"""

import collections
from collections import abc
import datetime
from io import StringIO
import itertools
from textwrap import dedent
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)
import warnings

import numpy as np
import numpy.ma as ma

if TYPE_CHECKING:
    from pandas.core.groupby.generic import DataFrameGroupBy
    from pandas.io.formats.style import Styler

# ---------------------------------------------------------------------
# Docstring templates

_shared_doc_kwargs = dict(
    axes="index, columns",
    klass="DataFrame",
    axes_single_arg="{0 or 'index', 1 or 'columns'}",
    axis="""axis : {0 or 'index', 1 or 'columns'}, default 0
        If 0 or 'index': apply function to each column.
        If 1 or 'columns': apply function to each row.""",
    optional_by="""
        by : str or list of str
            Name or list of names to sort by.

            - if `axis` is 0 or `'index'` then `by` may contain index
              levels and/or column labels.
            - if `axis` is 1 or `'columns'` then `by` may contain column
              levels and/or index labels.

            .. versionchanged:: 0.23.0

               Allow specifying index or column level names.""",
    versionadded_to_excel="",
    optional_labels="""labels : array-like, optional
            New labels / index to conform the axis specified by 'axis' to.""",
    optional_axis="""axis : int or str, optional
            Axis to target. Can be either the axis name ('index', 'columns')
            or number (0, 1).""",
)

_numeric_only_doc = """numeric_only : boolean, default None
    Include only float, int, boolean data. If None, will attempt to use
    everything, then use only numeric data
"""

_merge_doc = """
Merge DataFrame or named Series objects with a database-style join.

The join is done on columns or indexes. If joining columns on
columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
on indexes or indexes on a column or columns, the index will be passed on.

Parameters
----------%s
right : DataFrame or named Series
    Object to merge with.
how : {'left', 'right', 'outer', 'inner'}, default 'inner'
    Type of merge to be performed.

    * left: use only keys from left frame, similar to a SQL left outer join;
      preserve key order.
    * right: use only keys from right frame, similar to a SQL right outer join;
      preserve key order.
    * outer: use union of keys from both frames, similar to a SQL full outer
      join; sort keys lexicographically.
    * inner: use intersection of keys from both frames, similar to a SQL inner
      join; preserve the order of the left keys.
on : label or list
    Column or index level names to join on. These must be found in both
    DataFrames. If `on` is None and not merging on indexes then this defaults
    to the intersection of the columns in both DataFrames.
left_on : label or list, or array-like
    Column or index level names to join on in the left DataFrame. Can also
    be an array or list of arrays of the length of the left DataFrame.
    These arrays are treated as if they are columns.
right_on : label or list, or array-like
    Column or index level names to join on in the right DataFrame. Can also
    be an array or list of arrays of the length of the right DataFrame.
    These arrays are treated as if they are columns.
left_index : bool, default False
    Use the index from the left DataFrame as the join key(s). If it is a
    MultiIndex, the number of keys in the other DataFrame (either the index
    or a number of columns) must match the number of levels.
right_index : bool, default False
    Use the index from the right DataFrame as the join key. Same caveats as
    left_index.
sort : bool, default False
    Sort the join keys lexicographically in the result DataFrame. If False,
    the order of the join keys depends on the join type (how keyword).
suffixes : tuple of (str, str), default ('_x', '_y')
    Suffix to apply to overlapping column names in the left and right
    side, respectively. To raise an exception on overlapping columns use
    (False, False).
copy : bool, default True
    If False, avoid copy if possible.
indicator : bool or str, default False
    If True, adds a column to output DataFrame called "_merge" with
    information on the source of each row.
    If string, column with information on source of each row will be added to
    output DataFrame, and column will be named value of string.
    Information column is Categorical-type and takes on a value of "left_only"
    for observations whose merge key only appears in 'left' DataFrame,
    "right_only" for observations whose merge key only appears in 'right'
    DataFrame, and "both" if the observation's merge key is found in both.

validate : str, optional
    If specified, checks if merge is of specified type.

    * "one_to_one" or "1:1": check if merge keys are unique in both
      left and right datasets.
    * "one_to_many" or "1:m": check if merge keys are unique in left
      dataset.
    * "many_to_one" or "m:1": check if merge keys are unique in right
      dataset.
    * "many_to_many" or "m:m": allowed, but does not result in checks.

    .. versionadded:: 0.21.0

Returns
-------
DataFrame
    A DataFrame of the two merged objects.

See Also
--------
merge_ordered : Merge with optional filling/interpolation.
merge_asof : Merge on nearest keys.
DataFrame.join : Similar method using indices.

Notes
-----
Support for specifying index levels as the `on`, `left_on`, and
`right_on` parameters was added in version 0.23.0
Support for merging named Series objects was added in version 0.24.0

Examples
--------
>>> df1 = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
...                     'value': [1, 2, 3, 5]})
>>> df2 = pd.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
...                     'value': [5, 6, 7, 8]})
>>> df1
    lkey value
0   foo      1
1   bar      2
2   baz      3
3   foo      5
>>> df2
    rkey value
0   foo      5
1   bar      6
2   baz      7
3   foo      8

Merge df1 and df2 on the lkey and rkey columns. The value columns have
the default suffixes, _x and _y, appended.

>>> df1.merge(df2, left_on='lkey', right_on='rkey')
  lkey  value_x rkey  value_y
0  foo        1  foo        5
1  foo        1  foo        8
2  foo        5  foo        5
3  foo        5  foo        8
4  bar        2  bar        6
5  baz        3  baz        7

Merge DataFrames df1 and df2 with specified left and right suffixes
appended to any overlapping columns.

>>> df1.merge(df2, left_on='lkey', right_on='rkey',
...           suffixes=('_left', '_right'))
  lkey  value_left rkey  value_right
0  foo           1  foo            5
1  foo           1  foo            8
2  foo           5  foo            5
3  foo           5  foo            8
4  bar           2  bar            6
5  baz           3  baz            7

Merge DataFrames df1 and df2, but raise an exception if the DataFrames have
any overlapping columns.

>>> df1.merge(df2, left_on='lkey', right_on='rkey', suffixes=(False, False))
Traceback (most recent call last):
...
ValueError: columns overlap but no suffix specified:
    Index(['value'], dtype='object')
"""


# -----------------------------------------------------------------------
# DataFrame class


class DataFrame(NDFrame):
    """
    Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Data structure also contains labeled axes (rows and columns).
    Arithmetic operations align on both row and column labels. Can be
    thought of as a dict-like container for Series objects. The primary
    pandas data structure.

    Parameters
    ----------
    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        Dict can contain Series, arrays, constants, or list-like objects.

        .. versionchanged:: 0.23.0
           If data is a dict, column order follows insertion-order for
           Python 3.6 and later.

        .. versionchanged:: 0.25.0
           If data is a list of dicts, column order follows insertion-order
           for Python 3.6 and later.

    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    columns : Index or array-like
        Column labels to use for resulting frame. Will default to
        RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer.
    copy : bool, default False
        Copy data from inputs. Only affects DataFrame / 2d ndarray input.

    See Also
    --------
    DataFrame.from_records : Constructor from tuples, also record arrays.
    DataFrame.from_dict : From dicts of Series, arrays, or dicts.
    read_csv : Read a comma-separated values (csv) file into DataFrame.
    read_table : Read general delimited file into DataFrame.
    read_clipboard : Read text from clipboard into DataFrame.

    Examples
    --------
    Constructing DataFrame from a dictionary.

    >>> d = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = pd.DataFrame(data=d)
    >>> df
       col1  col2
    0     1     3
    1     2     4

    Notice that the inferred dtype is int64.

    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object

    To enforce a single dtype:

    >>> df = pd.DataFrame(data=d, dtype=np.int8)
    >>> df.dtypes
    col1    int8
    col2    int8
    dtype: object

    Constructing DataFrame from numpy ndarray:

    >>> df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    ...                    columns=['a', 'b', 'c'])
    >>> df2
       a  b  c
    0  1  2  3
    1  4  5  6
    2  7  8  9
    """

    _internal_names_set = {"columns", "index"} | NDFrame._internal_names_set
    _typ = "dataframe"

    @property
    def _constructor(self) -> Type["DataFrame"]:
        return DataFrame

    _constructor_sliced: Type[Series] = Series
    _deprecations: FrozenSet[str] = NDFrame._deprecations | frozenset([])
    _accessors: Set[str] = {"sparse"}

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError("Not supported for DataFrames!")

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        data=None,
        index: Optional[Axes] = None,
        columns: Optional[Axes] = None,
        dtype: Optional[Dtype] = None,
        copy: bool = False,
    ):
        if data is None:
            data = {}
        if dtype is not None:
            dtype = self._validate_dtype(dtype)

        if isinstance(data, DataFrame):
            data = data._data

        if isinstance(data, BlockManager):
            mgr = self._init_mgr(
                data, axes=dict(index=index, columns=columns), dtype=dtype, copy=copy
            )
        elif isinstance(data, dict):
            mgr = init_dict(data, index, columns, dtype=dtype)
        elif isinstance(data, ma.MaskedArray):
            import numpy.ma.mrecords as mrecords

            # masked recarray
            if isinstance(data, mrecords.MaskedRecords):
                mgr = masked_rec_array_to_mgr(data, index, columns, dtype, copy)

            # a masked array
            else:
                mask = ma.getmaskarray(data)
                if mask.any():
                    data, fill_value = maybe_upcast(data, copy=True)
                    data.soften_mask()  # set hardmask False if it was True
                    data[mask] = fill_value
                else:
                    data = data.copy()
                mgr = init_ndarray(data, index, columns, dtype=dtype, copy=copy)

        elif isinstance(data, (np.ndarray, Series, Index)):
            if data.dtype.names:
                data_columns = list(data.dtype.names)
                data = {k: data[k] for k in data_columns}
                if columns is None:
                    columns = data_columns
                mgr = init_dict(data, index, columns, dtype=dtype)
            elif getattr(data, "name", None) is not None:
                mgr = init_dict({data.name: data}, index, columns, dtype=dtype)
            else:
                mgr = init_ndarray(data, index, columns, dtype=dtype, copy=copy)

        # For data is list-like, or Iterable (will consume into list)
        elif isinstance(data, abc.Iterable) and not isinstance(data, (str, bytes)):
            if not isinstance(data, (abc.Sequence, ExtensionArray)):
                data = list(data)
            if len(data) > 0:
                if is_list_like(data[0]) and getattr(data[0], "ndim", 1) == 1:
                    if is_named_tuple(data[0]) and columns is None:
                        columns = data[0]._fields
                    arrays, columns = to_arrays(data, columns, dtype=dtype)
                    columns = ensure_index(columns)

                    # set the index
                    if index is None:
                        if isinstance(data[0], Series):
                            index = get_names_from_index(data)
                        elif isinstance(data[0], Categorical):
                            index = ibase.default_index(len(data[0]))
                        else:
                            index = ibase.default_index(len(data))

                    mgr = arrays_to_mgr(arrays, columns, index, columns, dtype=dtype)
                else:
                    mgr = init_ndarray(data, index, columns, dtype=dtype, copy=copy)
            else:
                mgr = init_dict({}, index, columns, dtype=dtype)
        else:
            try:
                arr = np.array(data, dtype=dtype, copy=copy)
            except (ValueError, TypeError) as err:
                exc = TypeError(
                    "DataFrame constructor called with "
                    f"incompatible data and dtype: {err}"
                )
                raise exc from err

            if arr.ndim == 0 and index is not None and columns is not None:
                values = cast_scalar_to_array(
                    (len(index), len(columns)), data, dtype=dtype
                )
                mgr = init_ndarray(
                    values, index, columns, dtype=values.dtype, copy=False
                )
            else:
                raise ValueError("DataFrame constructor not properly called!")

        NDFrame.__init__(self, mgr)
