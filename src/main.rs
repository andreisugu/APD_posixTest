use mpi::traits::*;

fn main() {
    // Initialize the MPI environment
    let universe = mpi::initialize().unwrap();
    
    // Get the default communicator
    let world = universe.world();

    // Get the total number of processes and the specific rank
    let size = world.size();
    let rank = world.rank();

    // Get the name of the processor/host
    let processor_name = mpi::environment::processor_name()
        .unwrap_or_else(|_| String::from("Unknown"));

    // Print the message from each process
    println!(
        "Hello from processor {}, rank {} out of {} processors",
        processor_name, rank, size
    );
}