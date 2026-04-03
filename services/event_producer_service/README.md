# Event Producer Service 

We developed this service for data aggregation and formatting


For creating generation python file you have to use this command
```
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. api/api.proto  
```