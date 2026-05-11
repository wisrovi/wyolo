from setproctitle import setproctitle
from wpipe import (
    Condition,
    Pipeline,
    ResourceMonitor,
    TaskTimer,
)

from states import (
    error_capture,
    check_dataset,
    check_gpu_available,
    check_minio_buckets,
    train_model,
    load_yaml,
)

setproctitle("wtrain-service")


def config_pipeline():
    pipeline = Pipeline(
        pipeline_name="wtrain_pipeline",
        verbose=False,
        tracking_db="wtrain.db",
        #
        # max_retries=3,  # Retry up to 3 times
        # retry_delay=0.5,  # Wait 0.5 seconds between retries
        # retry_on_exceptions=(RuntimeError,),  # Only retry on RuntimeError
        # #
        collect_system_metrics=True,  # Enable metrics collection
        #
        show_progress=False,
    )

    pipeline.add_error_capture([error_capture])

    if pipeline:
        pipeline.set_steps(
            [
                load_yaml,
                check_minio_buckets,
                check_gpu_available,
                check_dataset,
                Condition(
                    expression="gpu_status == 1 and dataset_status == 1",
                    branch_true=[train_model],
                    branch_false=[],
                ),
            ]
        )

    return pipeline


def main(user_config_train):
    pipeline = config_pipeline()

    # _user_config_file = get_argument_parser()

    # user_config_train = get_user_config(user_config_file=_user_config_file)

    args_dict = {
        "user_config_train": "/wyolo/control_server/datasets/clasification/colorball.v8i.multiclass/config_train.yaml",
    }


    with ResourceMonitor("eyesdcar_pipeline_ResourceMonitor") as monitor:
        with TaskTimer("eyesdcar_pipeline_TaskTimer", timeout_seconds=900) as timer:
            results = pipeline.run(args_dict)

            if timer.exceeded_timeout():
                # print("⚠ Work exceeded timeout!")
                pass
            else:
                # print("✓ Work completed within timeout")
                pass

    # Resumen de recursos al terminar
    print(f"\nResource Summary:")
    summary = monitor.get_summary()
    print(f"  - Peak RAM: {summary['peak_ram_mb']} MB")
    print(f"  - Avg CPU: {summary['avg_cpu_percent']}%")
    print(f"✓ Total time monitored: {timer.elapsed_seconds:.2f}s")

    # if "error" in results:
    #     print(f"Error detectado: {results.get('error')}")

    return results


if __name__ == "__main__":

    results = main(dict(user_config_train=True))
    print(f"\nPipeline Results: {results}")
    print("gpu_status:", results.get("gpu_status"))
    print("dataset_status:", results.get("dataset_status"))
