echo 'set5' &&
echo 'x2' &&
python test_ciaosr.py --config ./configs/ciaosr/test-set5-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test_ciaosr.py --config ./configs/ciaosr/test-set5-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test_ciaosr.py --config ./configs/ciaosr/test-set5-4.yaml --model $1 --gpu $2 &&

echo 'set14' &&
echo 'x2' &&
python test_ciaosr.py --config ./configs/ciaosr/test-set14-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test_ciaosr.py --config ./configs/ciaosr/test-set14-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test_ciaosr.py --config ./configs/ciaosr/test-set14-4.yaml --model $1 --gpu $2 &&

echo 'b100' &&
echo 'x2' &&
python test_ciaosr.py --config ./configs/ciaosr/test-b100-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test_ciaosr.py --config ./configs/ciaosr/test-b100-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test_ciaosr.py --config ./configs/ciaosr/test-b100-4.yaml --model $1 --gpu $2 &&

echo 'urban100' &&
echo 'x2' &&
python test_ciaosr.py --config ./configs/ciaosr/test-urban100-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
python test_ciaosr.py --config ./configs/ciaosr/test-urban100-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
python test_ciaosr.py --config ./configs/ciaosr/test-urban100-3.yaml --model $1 --gpu $2 &&

true
