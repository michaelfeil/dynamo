#!/usr/bin/env python3
"""
Simple load test for testing remote servers.
Focuses on generating load without memory profiling on client side.
"""

import asyncio
import time
import traceback

import aiohttp

PAYLOAD_SIZE = 100000


class LoadTestDebugger:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        # Use collections.deque with maxlen to prevent unbounded memory growth
        from collections import deque

        self.request_times = deque(maxlen=10000)  # Keep only last 10k request times
        self.errors = deque(maxlen=1000)  # Keep only last 1k errors
        self.total_requests = 0
        self.total_errors = 0

    async def start_session(self):
        """Start HTTP session"""
        self.session = aiohttp.ClientSession()

    async def stop_session(self):
        """Stop HTTP session"""
        if self.session:
            await self.session.close()

    async def test_connection(self):
        """Test if the backend is accessible"""
        try:
            async with self.session.get(
                f"{self.base_url}/health_file", timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    print(f"✓ Backend is accessible at {self.base_url}")
                    return True
                else:
                    print(f"✗ Backend returned status {response.status}")
                    return False
        except Exception as e:
            print(f"✗ Cannot connect to backend at {self.base_url}: {e}")
            return False

    async def send_request(self, request_id: int):
        """Send a single request and measure performance"""
        start_time = time.time()

        payload = {
            "model": "mock_model",
            "messages": [
                {
                    "role": "user",
                    "content": f"Hello from request {request_id}" * PAYLOAD_SIZE,
                }
            ],
            "max_tokens": 2,
            "request_id": str(request_id),
        }

        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    # Consume the streaming response without storing all chunks
                    chunk_count = 0
                    async for line in response.content:
                        if line.strip():
                            chunk_count += 1
                            # Don't store chunks to save memory - just count them
                            pass

                    end_time = time.time()
                    self.request_times.append(end_time - start_time)
                    self.total_requests += 1

                    if request_id % 1000 == 0:
                        print(
                            f"Request {request_id} completed in {end_time - start_time:.3f}s"
                        )
                        if self.request_times:
                            recent_times = (
                                list(self.request_times)[-100:]
                                if len(self.request_times) >= 100
                                else list(self.request_times)
                            )
                            print(
                                f"Average response time: {sum(recent_times) / len(recent_times):.3f}s"
                            )

                else:
                    self.errors.append(f"Request {request_id}: HTTP {response.status}")
                    self.total_errors += 1

        except Exception as e:
            self.errors.append(f"Request {request_id}: {str(e)}")
            self.total_errors += 1
            if self.total_errors % 100 == 0:
                print(f"Total error count: {self.total_errors}")

    async def run_load_test(self, total_requests: int = 100000, concurrent: int = 10):
        """Run load test"""
        print(
            f"Starting load test: {total_requests} requests with {concurrent} concurrent"
        )

        test_start_time = time.time()
        semaphore = asyncio.Semaphore(concurrent)

        async def bounded_request(request_id):
            async with semaphore:
                await self.send_request(request_id)

        # Use smaller batches and process more efficiently
        batch_size = 500  # Reduced batch size
        completed_requests = 0

        for batch_start in range(0, total_requests, batch_size):
            batch_end = min(batch_start + batch_size, total_requests)

            # Create and execute tasks for this batch
            tasks = [
                asyncio.create_task(bounded_request(i))
                for i in range(batch_start, batch_end)
            ]

            # Wait for this batch to complete
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                completed_requests += len(
                    [r for r in results if not isinstance(r, Exception)]
                )
            except Exception as e:
                print(f"Batch {batch_start}-{batch_end} error: {e}")

            # Explicitly clean up task references
            for task in tasks:
                if not task.done():
                    task.cancel()
                # Help GC by removing references
                del task
            del tasks

            # Progress update every few batches
            if batch_end % 5000 == 0:
                elapsed = time.time() - test_start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                print(
                    f"Progress: {batch_end}/{total_requests} requests, rate: {rate:.1f} req/s"
                )
                print(f"  Completed: {completed_requests}, Errors: {self.total_errors}")

        # Final analysis
        print("\nLoad test completed!")
        print(f"Total requests sent: {total_requests}")
        print(f"Successful requests: {self.total_requests}")
        print(f"Total errors: {self.total_errors}")
        if self.request_times:
            print(
                f"Average response time: {sum(self.request_times) / len(self.request_times):.3f}s"
            )
        else:
            print("No successful requests - all requests failed!")
            print("Check backend server is running and accessible")

        if self.errors:
            print(f"\nLast 10 errors (of {self.total_errors} total):")
            for error in list(self.errors)[-10:]:
                print(f"  {error}")

            # Show error summary
            error_types = {}
            for error in self.errors:
                error_type = error.split(":")[1].strip() if ":" in error else "unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1

            print(f"\nError summary (from last {len(self.errors)} errors):")
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count} occurrences")


async def main():
    debugger = LoadTestDebugger()

    try:
        await debugger.start_session()

        # Test connection first
        if not await debugger.test_connection():
            print(
                "Cannot connect to backend. Make sure it's running at http://localhost:8000"
            )
            return

        # Run a smaller test first
        print("Running initial test...")
        start_time = time.time()
        await debugger.run_load_test(total_requests=100, concurrent=10)

        # If that works, run the full test
        print("\nRunning full load test...")
        start_time = time.time()
        await debugger.run_load_test(total_requests=20000, concurrent=500)
        total_time = time.time() - start_time
        print(f"Total test time: {total_time:.2f}s")

    except KeyboardInterrupt:
        print("Load test interrupted")
    except Exception as e:
        print(f"Load test failed: {e}")
        traceback.print_exc()
    finally:
        await debugger.stop_session()


if __name__ == "__main__":
    asyncio.run(main())
