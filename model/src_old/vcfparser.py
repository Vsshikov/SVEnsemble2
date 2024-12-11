import time
import re
import gzip
import builtins
import pandas as pd
import numpy as np

# BASIC READING FUNCTIONS

class VCFMetaEntry(dict):
    """
    Class for processing VCF metadata entries so they can be used for parsing the main file contents
    """
    
    meta_id = None
    number = None
    desc = None
    type = None

    
    def __init__(self, field):
        self.field = field


def get_read_params(file_path, force_format='infer'):
    """
    Checks the supplied path for file extensions and returns a tibble containing:
    - the module from which .open function will be called (may change it later)
    - the read mode string for that .open function
    """
    
    if force_format == 'vcf':
        return builtins, 'r'

    is_gzip = re.search(r'\.(gz|gzip)$', file_path)

    if is_gzip:
        vcf_file_handler_mod = gzip
        readmode = 'rt'
    else:
        vcf_file_handler_mod = builtins
        readmode = 'r'

    return vcf_file_handler_mod, readmode


def read_header_line(line, info_limit_to='all'):
    """
    Reads an unprocessed line from VCF header and returns a VCFHeaderEntry object
    """

    try:
        entry = VCFMetaEntry(re.search(r'##[A-Z]+=', line).group().replace("##", "").replace("=", ""))
        entry.meta_id = re.search(r'ID=([A-Z]|[a-z]|[0-9]|_)+', line).group().replace("ID=", "")
    except AttributeError:
        return None

    try:
        number_val = int(re.search(r'Number=([0-9]|\.)+,', line).group().replace("Number=", "").replace(",", ""))
        entry.number = number_val
    except ValueError:
        number_val = re.search(r'Number=([0-9]|\.)+,', line).group().replace("Number=", "").replace(",", "")
        entry.number = number_val
    except AttributeError:
        entry.number = None

    try:
        type_val = re.search(r'Type=([A-Z]|[a-z]|[0-9])+,', line).group().replace("Type=", "").replace(",", "")
        entry.type = type_val
    except AttributeError:
        entry.type = None

    try:
        entry.desc = re.search(r'Description=".+"', line).group().replace('Description=\"', '').replace('"', '')
    except AttributeError:
        entry.desc = None
    
    return entry

def process_meta_entry(vcf_dict, meta_dict, meta_entry, info_limit_to='all'):
    """
    Accepts a VCFMetaEntry object along with 2 dictionaries \
    - first dict is to be used for storing the full vcf data
    - second dict is to be used for storing vcf metadata

    Updates the dictionaries with information from the meta entry
    - adds an {entry field}_{entry id}: [] item to the vcf data dict
    - adds an {entry field}_{entry id}: (entry number, entry value type, entry description)\
    item to the meta dict. Note that meta dict is a nested dictionary \
    (it is a dict with FILTER, FORMAT, and INFO fields, all of which are used\
    to address a dictionary with corresponding meta entry IDs as keys), 
    """
    if meta_entry is not None:
        if meta_entry.field in meta_dict:
            if info_limit_to != 'all':
                if (meta_entry.field == 'INFO') and (meta_entry.meta_id not in info_limit_to):
                    return
            
            meta_dict[meta_entry.field].update(
                {
                    f'{meta_entry.field}_{meta_entry.meta_id}':\
                    (meta_entry.number, meta_entry.type, meta_entry.desc)
                }
            )
            # print(f'[{meta_entry.field}] {meta_entry.meta_id}, number: {meta_entry.number}, type: {meta_entry.type}, desc: {meta_entry.desc}')
            
            vcf_dict.update({f'{meta_entry.field}_{meta_entry.meta_id}': []})


def process_filter_entry(vcf_dict, meta_dict, filter_str, filter_delim=';'):
    """
    Accepts a vcf dict and meta dict along with a filter entry string.
    Iterates over FILTER entries in the meta dict.
    If any of the FILTER_{filter entry id} appear in the string, \
    appends the corresponding list in the vcf dict with 1
    Otherwise, appends the corresponding list with 0
    """
    filter_entry_with_prefix = [f'FILTER_{x}' for x in filter_str.split(filter_delim)]
    # print(filter_entry_with_prefix)
    
    for filter_field in meta_dict['FILTER']:
        if filter_field in filter_entry_with_prefix:
            vcf_dict[filter_field].append(1)
        else:
            vcf_dict[filter_field].append(0)


def process_info_entry(vcf_dict, meta_dict, info_str, info_delim=';', info_eq='=', limit_to='all'):
    """
    Accepts a vcf dict and meta dict along with an info entry string.
    NOTE TO SELF - COULD PROB OPTIMIZE

    Splits the info string based on the supplied delimeter and equal sign
    Iterates over the resulting list and appends lists in the vcf dict with corresponding values
    If any field from vcf meta is not present in the string, \
    appends corresponding list in vcf dict with 0
    """

    if limit_to == 'all':
        info_dict = {}
        for info_item_raw in info_str.split(info_delim):
            info_item = info_item_raw.split(info_eq)
            if len(info_item) == 1:
                info_dict.update({f'INFO_{info_item[0]}': 1})
            elif len(info_item) == 2:
                info_dict.update({f'INFO_{info_item[0]}': info_item[1]})
            else:
                raise IndexError('INFO field entry with wrong syntaxis detected')
    else:
        info_dict = {}
        n_matched = 0
        for info_item_raw in info_str.split(info_delim):
            if n_matched >= len(limit_to):
                break
            info_item = info_item_raw.split(info_eq)
            if info_item[0] not in limit_to:
                continue
            n_matched += 1
            if len(info_item) == 1:
                info_dict.update({f'INFO_{info_item[0]}': 1})
            elif len(info_item) == 2:
                info_dict.update({f'INFO_{info_item[0]}': info_item[1]})
            else:
                raise IndexError('INFO field entry with wrong syntaxis detected')
    
    
    info_fields = info_dict.keys()
    for info_field in meta_dict['INFO']:
        if info_field in info_fields:
            vcf_dict[info_field].append(info_dict[info_field])
        else:
            vcf_dict[info_field].append(0)


def process_genotype_entry(vcf_dict, meta_dict, gt_list, format_delim=':'):
    """
    Accepts a vcf dict and meta dict. Unlike other similar function, accepts genotype info \
    as a list of strings.

    Currently only supports a single genotype (FORMAT str + 1 genotype str)

    Iterates over the format fields and appends values to the corresponding vcf dict lists
    """
    format_dict = {
        f'FORMAT_{x[0]}': x[1] for x in
        zip(gt_list.pop(0).split(format_delim), gt_list.pop(0).split(format_delim))
    }

    format_fields = format_dict.keys()
    for format_field in meta_dict['FORMAT']:
        if format_field in format_fields:
            vcf_dict[format_field].append(format_dict[format_field])
        else:
            vcf_dict[format_field].append(0)


def parse_vcf(
    vcf_path,
    filter_delim=';',
    info_delim=';', info_eq='=',
    format_present='infer', format_delim=':',
    line_limit=None, report_freq=100000,
    output_meta=False, only_filter_pass=False, skip_info=False, skip_format=False,
    info_limit_to='all'
):
    """
    Reads the vcf file specified in vcf_path argument and converts it into pandas dataframe.
    Uses information from the header, so will not work if the file does not have it.

    If output_meta=False (default) outputs a dataframe, if output_meta=True returns a tuple (vcf_dataframe, meta_dict)
    """

    if line_limit is None:
        line_limit = 0
    
    meta_dict = {
        'FILTER': {},
        'INFO': {},
        'FORMAT': {}
    }
    
    vcf_data = {
        'CHROM': [],
        'POS': [],
        'ID': [],
        'REF': [],
        'ALT': [],
        'QUAL': []
    }

    vcf_file_handler_mod, readmode = get_read_params(vcf_path, force_format=format_present)
    
    header_nlines = 0
    with vcf_file_handler_mod.open(vcf_path, readmode) as vcf:
        
        start_time = time.time()
        i = 0
        for i, line in enumerate(vcf):
            if i % report_freq == 0:
                current_time = time.time()
                print(f'Processed {i} entries \
                Elapsed time: {current_time - start_time} seconds.')
            
            # There are 2 major blocks processed differently: the header and the body
            # Currently the algorithm to process a headerless vcf file is not done, so a header is mandatory
    
            if line.startswith('#'):
                header_nlines += 1
                # Both the metadata and the header itself has a # / ## in the beginning
                if line.startswith('##'):
                    # print(line)
                    meta_entry = read_header_line(line)
                    if meta_entry is None:
                        continue
                    if not skip_info or meta_entry.field != 'INFO':
                        process_meta_entry(vcf_data, meta_dict, meta_entry, info_limit_to=info_limit_to)
                else:
                    continue
            else:
                if (i - header_nlines >= line_limit) & (line_limit != 0):
                    # print(vcf_data)
                    # print(meta_dict)
                    break
                
                line_list = line.split('\t')
                # print(line_list)
    
                # First fields are easily processed
                vcf_data['CHROM'].append(line_list.pop(0))
                vcf_data['POS'].append(line_list.pop(0))
                vcf_data['ID'].append(line_list.pop(0))
                vcf_data['REF'].append(line_list.pop(0))
                vcf_data['ALT'].append(line_list.pop(0))
                vcf_data['QUAL'].append(line_list.pop(0))
    
                # FILTER
                # The filter field is referenced to all possible filter field in the metadata
                # If an entry is present, then the column is saved with value 1, else - with 0
                process_filter_entry(
                    vcf_data, meta_dict,
                    filter_str=line_list.pop(0),
                    filter_delim=filter_delim
                )
    
                # INFO
                # Here the program iterates over the possible INFO field values listed in the header too
                # info_entry_with_prefix_raw = [f'INFO_{x}' for x in line_list.pop(0).split(info_delim)]
                if not skip_info:
                    process_info_entry(
                        vcf_data, meta_dict,
                        info_str=line_list.pop(0),
                        info_delim=info_delim, info_eq=info_eq,
                        limit_to=info_limit_to
                    )
                else:
                    line_list.pop(0)  # Skip the INFO field

                # FORMAT and genotype columns
                # Currently only a single genotype column is supported
                if skip_format:
                    continue

                process_genotype_entry(
                    vcf_data, meta_dict,
                    line_list,
                    format_delim=format_delim
                )
                
    
    keys_for_deletion = []
    # max_len = max([len(item) for item in vcf_data.items()])
    for vcf_data_key, vcf_data_item in vcf_data.items():
        # print(f'{vcf_data_key}: {len(vcf_data_item)}')
        if len(vcf_data_item) == 0:
            keys_for_deletion.append(vcf_data_key)
    if len(keys_for_deletion) > 0:
        print(f'Warning: items {", ".join(keys_for_deletion)} present in header but not present in data.')
    for key_for_deletion in keys_for_deletion:
        del vcf_data[key_for_deletion]
        # Small fix so an entire section does not get deleted if INFO/FORMAT/FILTER
        # is mentioned in the metadata
        if key_for_deletion in ['INFO', 'FORMAT', 'FILTER']:
            continue
        try:
            del meta_dict[re.match(r'(INFO)|(FORMAT)|(FILTER)', key_for_deletion).group()][key_for_deletion]
        except KeyError:
            print(f'Unexpected meta ID: {key_for_deletion}')
        # try:
        #     del meta_dict['INFO'][key_for_deletion]
        #     del meta_dict['FORMAT'][key_for_deletion]
        #     del meta_dict['FILTER'][key_for_deletion]
        # except KeyError:
        #     continue


    # Remove empty fields
    vcf_data = pd.DataFrame(vcf_data)

    try:
        vcf_data['FILTER_PASS']
    except KeyError:
        filter_cols = [col for col in vcf_data.columns if col.startswith('FILTER_') and not col.startswith('FILTER_PASS')]
        # print(np.logical_and.reduce(df2_copy[filter_cols2] == 1, axis=0))
        vcf_data.loc[:, 'FILTER_PASS'] = np.logical_and.reduce(vcf_data[filter_cols] == 0, axis=1)
        meta_dict['FILTER'].update({'FILTER_PASS': [None, 'flag', 'Passing all filters']})

    vcf_data.POS = pd.to_numeric(vcf_data.POS)
    vcf_data.QUAL = pd.to_numeric(vcf_data.QUAL, errors='coerce')
    
    current_time = time.time()
    print(f'Finished processing. Processed {i - header_nlines} entries out of {len(vcf_data)}. Elapsed time: {current_time - start_time} seconds.')

    if output_meta:
        return vcf_data, meta_dict
        
    return vcf_data

