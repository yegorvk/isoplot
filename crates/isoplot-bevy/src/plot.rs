use std::sync::Arc;

use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    pbr::wireframe::Wireframe,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
};
use bytemuck::cast_vec;
use dashmap::{DashMap, mapref::one::Ref};
use glam::IVec3;
use isoplot_mesh::{
    CentralDifference, ScalarField, SeparateNormals,
    dual_contouring::{BorrowChunk, Chunk, DualContouring},
};

type ExtractChunkFn = Arc<dyn (Fn(IVec3) -> Option<Mesh>) + Send + Sync>;

#[derive(Component)]
pub struct Plot {
    extract: ExtractChunkFn,
    render_distance: u32,
}

impl Plot {
    pub fn new<S>(source: S, render_distance: u32, max_level: u8, epsilon: f32) -> Self
    where
        S: ScalarField + Send + Sync + 'static,
    {
        let world = World::default();

        let extract = move |coords: IVec3| -> Option<Mesh> {
            let source = CentralDifference::new((&source).translated(coords.as_vec3()), epsilon);
            let mut sink = SeparateNormals::default();

            let dc = DualContouring::new(source, max_level);
            let chunk = dc.extract_chunk(&mut sink).ok()?;

            let peek = |offset: IVec3| world.get(coords + offset);
            dc.extract_seam(&chunk, peek, &mut sink).ok()?;

            world.insert(coords, chunk);
            (!sink.positions.is_empty()).then(|| build_bevy_mesh(sink))
        };

        Self {
            extract: Arc::new(extract),
            render_distance,
        }
    }
}

pub struct PlotPlugin;

impl Plugin for PlotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (queue_chunk_meshes, spawn_chunk_meshes));
    }
}

#[derive(Component, Debug)]
struct PlotChunk;

#[derive(Component)]
struct ExtractTask(Task<Vec<(IVec3, Mesh)>>);

fn queue_chunk_meshes(mut commands: Commands, query: Query<(Entity, &Plot), Added<Plot>>) {
    let pool = AsyncComputeTaskPool::get();

    for (entity, plot) in &query {
        let extract = plot.extract.clone();
        let r = plot.render_distance as i32;

        let task = pool.spawn(async move {
            let mut chunks = Vec::new();

            for x in -r..=r {
                for y in -r..=r {
                    for z in -r..=r {
                        let coords = IVec3::new(x, y, z);

                        if let Some(mesh) = extract(coords) {
                            chunks.push((coords, mesh));
                        }
                    }
                }
            }

            chunks
        });

        commands.entity(entity).insert(ExtractTask(task));
    }
}

fn spawn_chunk_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut query: Query<(Entity, &mut ExtractTask, &MeshMaterial3d<StandardMaterial>)>,
) {
    for (entity, mut task, material) in &mut query {
        let Some(chunks) = block_on(future::poll_once(&mut task.0)) else {
            continue;
        };

        for (coords, mesh) in chunks {
            commands.entity(entity).with_child((
                PlotChunk,
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(material.0.clone()),
                Wireframe,
                Transform::from_xyz(coords.x as f32, coords.y as f32, coords.z as f32),
            ));
        }

        commands.entity(entity).remove::<ExtractTask>();
    }
}

fn build_bevy_mesh(data: SeparateNormals) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals);
    mesh.insert_indices(Indices::U32(cast_vec(data.indices)));

    mesh
}

#[derive(Debug, Default)]
struct World {
    chunks: DashMap<IVec3, Chunk>,
}

impl World {
    fn get(&self, coords: IVec3) -> Option<ChunkGuard<'_>> {
        self.chunks.get(&coords).map(ChunkGuard)
    }

    fn insert(&self, coords: IVec3, chunk: Chunk) -> Option<Chunk> {
        self.chunks.insert(coords, chunk)
    }
}

struct ChunkGuard<'a>(Ref<'a, IVec3, Chunk>);

impl BorrowChunk for ChunkGuard<'_> {
    fn borrow_chunk(&self) -> &Chunk {
        self.0.value()
    }
}
