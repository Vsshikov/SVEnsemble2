"""
Legacy FASTA/FASTQ reader: Heng Li's readfq generator.

Unused. Kept from the removed delly2bnd script (which used it to fetch the
reference base at Delly mate breakends) for a future sequence fetcher, e.g.
filling REF/ALT anchor bases when writing VCFs. Prefer an indexed reader
(pysam.FastaFile) for random access; this one streams the whole file.

Usage:
    with open(path) as fp:  # or gzip.open(path, 'rt') for .gz
        for name, seq, qual in readfq(fp):  # qual is None for FASTA
            ...
"""


def readfq(fp): # this is a generator function
    last = None # this is a buffer keeping the last unprocessed line
    while True: # mimic closure; is it a bad idea?
        if not last: # the first record or a record following a fastq
            for l in fp: # search for the start of the next record
                if l[0] in '>@': # fasta/q header line
                    last = l[:-1] # save this line
                    break
        if not last: break
        name, seqs, last = last[1:].partition(" ")[0], [], None
        for l in fp: # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+': # this is a fasta record
            yield name, ''.join(seqs), None # yield a fasta record
            if not last: break
        else: # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp: # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq): # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs); # yield a fastq record
                    break
            if last: # reach EOF before reading enough quality
                yield name, seq, None # yield a fasta record instead
                break
