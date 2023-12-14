#!/usr/bin/env bash
make clean build

PART_ID=Colin
kernels=(global shared register constant)
threadsPerBlock=128

for kernel in global shared register constant
do
  for elementsPerThread in 1 2 4 8 16 32 64
  do
      for iteration in 1 2 3 4 5 6 7 8
      do
        echo "partId: $PART_ID elements: $elementsPerThread threads: $threadsPerBlock kernel: $kernel"
        make run ARGS="-p $PART_ID -m $elementsPerThread -t $threadsPerBlock -k $kernel" >> output.txt
      done
  done
done

python output_data_parser.py output.csv