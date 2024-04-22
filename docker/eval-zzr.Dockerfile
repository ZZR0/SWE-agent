FROM sweagent/swe-agent:latest

RUN pip install git+https://github.com/ZZR0/SWE-bench.git@zzr
RUN pip install unidiff
