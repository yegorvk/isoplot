use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    platform::collections::HashMap,
    prelude::*,
};
use bytemuck::cast_vec;
use isoplot_mesh::{
    CentralDifference, ExtractError, ScalarField, SeparateNormals, TranslateMesh,
    dual_contouring::{Chunk, DualContouring},
};

#[derive(Component)]
pub struct Plot {
    field: Box<dyn ScalarField + Send + Sync>,
    render_distance: u8,
}

impl Plot {
    pub fn new<S>(field: S, render_distance: u8) -> Self
    where
        S: ScalarField + Send + Sync + 'static,
    {
        Plot {
            field: Box::new(field),
            render_distance,
        }
    }

    fn build_mesh_data(&self) -> Result<SeparateNormals, ExtractError> {
        let mut world = World::default();
        let r = self.render_distance as i32;

        let mut sink = SeparateNormals::default();

        for x in -r..=r {
            for y in -r..=r {
                for z in -r..=r {
                    let i_offset = glam::ivec3(x, y, z);
                    let offset = i_offset.as_vec3();

                    let source =
                        CentralDifference::new(self.field.as_ref(), 1e-4).translated(offset);

                    let dc = DualContouring::new(&source, 7);

                    let mut chunk_sink = TranslateMesh::new(&mut sink, offset);
                    let chunk = dc.extract_chunk(&mut chunk_sink)?;

                    dc.extract_seam(
                        &chunk,
                        |offset| world.chunks.get(&(i_offset + offset)),
                        &mut chunk_sink,
                    )?;

                    world.chunks.insert(i_offset, chunk);
                }
            }
        }

        Ok(sink)
    }
}

pub struct PlotPlugin;

impl Plugin for PlotPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, create_mesh);
    }
}

fn create_mesh(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    query: Query<(Entity, &Plot), Added<Plot>>,
) {
    for (entity, plot) in query {
        let Ok(data) = plot.build_mesh_data() else {
            error!("Failed to extract mesh.");
            continue;
        };

        let mesh = build_bevy_mesh(data);
        commands.entity(entity).insert(Mesh3d(meshes.add(mesh)));
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

#[derive(Default)]
struct World {
    chunks: HashMap<glam::IVec3, Chunk>,
}
