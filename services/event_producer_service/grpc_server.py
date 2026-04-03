import asyncio
import grpc
from grpc.aio import server
from concurrent import futures
import api.api_pb2 as api_pb2
import api.api_pb2_grpc as api_pb2_grpc
from models import CSVRow, TMSMessage, WMSMessage, YMSMessage
from broker_setup import tms_publisher, wms_publisher, yms_publisher, broker
from csv_parser import parse_csv_stream

class CSVUploadServicer(api_pb2_grpc.CSVUploadServiceServicer):
    async def UploadCSV(self, request_iterator, context):
        rows_processed = 0
        try:
            # Asynchronously iterate over each parsed CSV row
            async for raw_row in parse_csv_stream(request_iterator):
                # Validate and transform the raw row into the CSVRow model
                try:
                    csv_row = CSVRow(**raw_row)
                except Exception as e:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Invalid CSV row data: {str(e)}")
                    return api_pb2.UploadStatus(
                        success=False,
                        message=f"Invalid CSV row data: {str(e)}"
                    )

                # Transform the CSV row into messages for the three services
                tms_msg = TMSMessage(
                    office_id=csv_row.office_from_id,
                    period_start=csv_row.period_start,
                    trucks_to_call=csv_row.trucks_to_call,
                    status=csv_row.status
                )
                wms_msg = WMSMessage(
                    office_id=csv_row.office_from_id,
                    period_start=csv_row.period_start,
                    forecasted_demand=csv_row.forecasted_demand,
                    capacity_provided=csv_row.capacity_provided,
                    fill_rate_pct=csv_row.fill_rate_pct
                )
                yms_msg = YMSMessage(
                    office_id=csv_row.office_from_id,
                    period_start=csv_row.period_start,
                    shortage=csv_row.shortage,
                    excess_capacity=csv_row.excess_capacity
                )

                # Publish messages to their respective brokers asynchronously
                # Use asyncio.gather to publish all three in parallel for better performance
                await asyncio.gather(
                    tms_publisher.publish(tms_msg),
                    wms_publisher.publish(wms_msg),
                    yms_publisher.publish(yms_msg)
                )

                # print(tms_msg)
                # print(wms_msg)
                # print(yms_msg)


                rows_processed += 1

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return api_pb2.UploadStatus(
                success=False,
                message=f"Internal server error: {str(e)}"
            )

        return api_pb2.UploadStatus(
            success=True,
            message="CSV file processed successfully",
            rows_processed=rows_processed
        )

async def serve():
    # Start the FastStream application (this connects to the broker)
    await broker.start()
    # Create and start the gRPC server
    grpc_server = grpc.aio.server()
    api_pb2_grpc.add_CSVUploadServiceServicer_to_server(
        CSVUploadServicer(), grpc_server
    )
    grpc_server.add_insecure_port("[::]:50051")
    await grpc_server.start()
    print("gRPC server is running on port 50051...")
    await grpc_server.wait_for_termination()