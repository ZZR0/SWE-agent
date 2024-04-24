python run_docker_builder.py --image_name zzr/swe-env:latest
docker images | grep '^zzr/swe-env--' | awk '{print $1 ":" $2}' | xargs -I {} docker rmi {}
docker images | grep '<none>' | awk '{print $3}' | xargs -I {} docker rmi {}
docker build --no-cache --network host --build-arg all_proxy=http://192.168.100.211:10809 -t zzr/swe-env:latest -f docker/swe-env.Dockerfile .

python scripts/run_with_image.py --data_path /hdd2/zzr/SWE-agent/dataset/swebench_lite_dev.json
python scripts/valid_with_image.py --data_path /hdd2/zzr/SWE-agent/dataset/swebench_lite_dev.json