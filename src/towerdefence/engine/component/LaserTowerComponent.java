package towerdefence.engine.component;

import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.state.StateBasedGame;

import towerdefence.engine.ResourceManager;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Entity;
import towerdefence.engine.entity.Tower;

public class LaserTowerComponent extends RenderComponent{

	Tower tower;
	
	int shootingCounter;
	
	public LaserTowerComponent(String id) {
		super(id);
	}
	
	public void setOwnerEntity(Entity entity)
    {
        this.entity = entity;
        tower = (Tower)entity;
    }
	
	@Override
	public void update(GameContainer gc, StateBasedGame sb, int delta) {

		shootingCounter-=delta;
		
		if (tower.isActive() && shootingCounter <= 0 && 
		tower.getTargetCritter()!=null) {
            shootCritter(tower.getTargetCritter());
            shootingCounter = 100; 
        }
		
	}
	
	@Override
	public void render(GameContainer gc, StateBasedGame sb, Graphics gr) {
		
		switch (tower.getType()) {
		case Tower.NORMAL:
			gr.setColor(Color.green);
			break;
		case Tower.FIRE:
			gr.setColor(Color.red);
			break;
		case Tower.ICE:
			gr.setColor(Color.blue);
			break;
		default:
			gr.setColor(Color.green);
			break;
		}
		
		if (tower.getTargetCritter() != null) {

			gr.drawLine(tower.getPosition().x + 16, tower.getPosition().y + 16,
					tower.getTargetCritter().getPosition().x + 16, tower.getTargetCritter()
							.getPosition().y + 16);
		}
		
		gr.setColor(Color.white);
	}
	
    private void shootCritter(Critter critter) {
        if(critter.getType()==Critter.FIRE && tower.getType()==Tower.ICE) {
            critter.takeDamage((tower.getDamagePerSec()*2.0f)/10f);   
        } else if(critter.getType()==Critter.ICE && tower.getType()==Tower.FIRE) {
            critter.takeDamage((tower.getDamagePerSec()*1.5f)/10f);   
        } else if(tower.getType()==Tower.NORMAL && (critter.getType()!=Critter.NORMAL)){
        	critter.takeDamage(((tower.getDamagePerSec()*0.85f)/10f));
        } else {
            critter.takeDamage(tower.getDamagePerSec()/10f);
        }
        
    }

}
