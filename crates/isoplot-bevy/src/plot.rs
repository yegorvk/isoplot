use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use bytemuck::cast_vec;
use isoplot_mesh::{DualContouring, ExtractError, NormalField, SeparateNormals};

#[derive(Component)]
pub struct Plot {
    source: Box<dyn NormalField + Send + Sync>,
}

impl Plot {
    pub fn new<S>(source: S) -> Self
    where
        S: NormalField + Send + Sync + 'static,
    {
        Plot {
            source: Box::new(source),
        }
    }

    fn build_mesh_data(&self) -> Result<SeparateNormals, ExtractError> {
        Ok(DualContouring::new(self.source.as_ref()).extract()?)
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
